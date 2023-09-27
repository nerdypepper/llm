//! An implementation of [LLaMA](https://huggingface.co/docs/transformers/model_doc/llama) for the `llm` ecosystem.
#![deny(missing_docs)]

use std::{error::Error, sync::Arc};

use llm_base::{
    ggml,
    model::{common, HyperparametersWriteError},
    util, FileType, GraphOutputs, InferenceSession, InferenceSessionConfig, KnownModel, LoadError,
    ModelParameters, OutputRequest, Regex, TensorLoader, TokenId, Tokenizer,
};
use num_traits::ToPrimitive;

/// The BERT model.
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Bert {
    params: ModelParameters,
    hyperparameters: Hyperparameters,
    tokenizer: Tokenizer,

    word_embeddings: ggml::Tensor,
    token_type_embeddings: ggml::Tensor,
    position_embeddings: ggml::Tensor,
    ln_e_w: ggml::Tensor,
    ln_e_b: ggml::Tensor,

    // weights for the model
    layers: Vec<Layer>,

    // must be kept alive for the model
    context: Arc<ggml::Context>,
}

unsafe impl Send for Bert {}
unsafe impl Sync for Bert {}

/// BERT [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// Size of the model's vocabulary
    pub n_vocab: usize,

    /// Maximum number of tokens
    pub n_max_tokens: usize,

    /// Size of the model's embedding layer
    pub n_embd: usize,

    /// n_head
    pub n_intermediate: usize,

    /// Number of attention heads
    pub n_head: usize,

    /// Number of layers in the model
    pub n_layer: usize,

    /// file_type
    pub file_type: FileType,
}

impl KnownModel for Bert {
    type Hyperparameters = Hyperparameters;

    fn new<E: Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        tokenizer: Tokenizer,
        tensor_loader: impl TensorLoader<E>,
    ) -> Result<Self, E> {
        let mut tl = tensor_loader;

        let word_embeddings = tl.load("embeddings.word_embeddings.weight")?;
        let token_type_embeddings = tl.load("embeddings.token_type_embeddings.weight")?;
        let position_embeddings = tl.load("embeddings.position_embeddings.weight")?;

        let ln_e_w = tl.load("embeddings.LayerNorm.weight")?;
        let ln_e_b = tl.load("embeddings.LayerNorm.bias")?;

        let mut layers = Vec::new();

        for i in 0..hyperparameters.n_layer {
            let backend = params.backend(i);

            let layer = Layer {
                ln_att_w: tl
                    .load(&format!(
                        "encoder.layer.{i}.attention.output.LayerNorm.weight"
                    ))?
                    .transfer_to(backend),
                ln_att_b: tl
                    .load(&format!(
                        "encoder.layer.{i}.attention.output.LayerNorm.bias"
                    ))?
                    .transfer_to(backend),

                // attention
                q_w: tl
                    .load(&format!("encoder.layer.{i}.attention.self.query.weight"))?
                    .transfer_to(backend),
                q_b: tl
                    .load(&format!("encoder.layer.{i}.attention.self.query.bias"))?
                    .transfer_to(backend),
                k_w: tl
                    .load(&format!("encoder.layer.{i}.attention.self.key.weight"))?
                    .transfer_to(backend),
                k_b: tl
                    .load(&format!("encoder.layer.{i}.attention.self.key.bias"))?
                    .transfer_to(backend),
                v_w: tl
                    .load(&format!("encoder.layer.{i}.attention.self.value.weight"))?
                    .transfer_to(backend),
                v_b: tl
                    .load(&format!("encoder.layer.{i}.attention.self.value.bias"))?
                    .transfer_to(backend),

                o_w: tl
                    .load(&format!("encoder.layer.{i}.attention.output.dense.weight"))?
                    .transfer_to(backend),
                o_b: tl
                    .load(&format!("encoder.layer.{i}.attention.output.dense.bias"))?
                    .transfer_to(backend),

                // ff
                ff_i_w: tl
                    .load(&format!("encoder.layer.{i}.intermediate.dense.weight"))?
                    .transfer_to(backend),
                ff_i_b: tl
                    .load(&format!("encoder.layer.{i}.intermediate.dense.bias"))?
                    .transfer_to(backend),

                ln_out_w: tl
                    .load(&format!("encoder.layer.{i}.output.LayerNorm.weight"))?
                    .transfer_to(backend),
                ln_out_b: tl
                    .load(&format!("encoder.layer.{i}.output.LayerNorm.bias"))?
                    .transfer_to(backend),
                ff_o_w: tl
                    .load(&format!("encoder.layer.{i}.output.dense.weight"))?
                    .transfer_to(backend),
                ff_o_b: tl
                    .load(&format!("encoder.layer.{i}.output.dense.bias"))?
                    .transfer_to(backend),
            };

            layers.push(layer);
        }
        let context = tl.finish();

        Ok(Self {
            ln_e_b,
            ln_e_w,
            position_embeddings,
            token_type_embeddings,
            word_embeddings,
            hyperparameters,
            params,
            tokenizer,
            layers,
            context: Arc::new(context),
        })
    }

    /// Starts a new `InferenceSession` for this model.
    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        InferenceSession::new(
            config,
            &self.params,
            self.hyperparameters.n_layer,
            self.hyperparameters.n_embd,
            self.hyperparameters.n_vocab,
        )
    }

    #[tracing::instrument(level = "trace", skip_all)]
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    ) {
       let  example_len: usize = input_tokens
            .iter()
            .map(|&i|  if i == self.pad_token_id().unwrap() {
                0
            } else {
                1
            }).sum();

            
        let input_tokens = &input_tokens[..example_len]; 
        let input_len = input_tokens.len();
        let _session_len = session.n_past;
        let _ctx_size = self.params.context_size;

        let Hyperparameters {
            n_vocab,
            n_max_tokens: _,
            n_embd,
            n_intermediate: _,
            n_head,
            n_layer,
            file_type: _,
        } = self.hyperparameters;

        let d_head = n_embd / n_head;

        let outputs = session.compute(self.context.clone(), input_tokens, |builder| {
            let mut ctx0 = builder.ctx0.borrow_mut();
            let gf = ggml::ComputationGraph::new();

            let embd = builder.embd;

            let mut input_layer = ctx0.op_get_rows(&self.word_embeddings, embd);

            // IL = word_embeddings + token_types + position_embeddingso
            {
                // token-types: a zero tensor
                let mut token_types = ctx0.new_tensor_1d(llm_base::ElementType::I32, input_len);
                token_types.zero_data();

                // position embeddings: another tensor
                let position_buf: Vec<i32> = (0..input_len as i32).collect();
                let mut positions = ctx0.new_tensor_1d(llm_base::ElementType::I32, input_len);
                unsafe { positions.write_data(bytemuck::cast_slice(&position_buf)) };

                // IL += token_types
                input_layer = ctx0.op_add(
                    &input_layer,
                    &ctx0.op_get_rows(&self.token_type_embeddings, &token_types),
                );

                // IL += position_embeddings
                input_layer = ctx0.op_add(
                    &input_layer,
                    &ctx0.op_get_rows(&self.position_embeddings, &positions),
                );
            }

            // embd norm
            {
                input_layer = ctx0.op_norm(&input_layer);
                input_layer = ctx0.op_add(&ctx0.op_mul(&input_layer, &self.ln_e_w), &self.ln_e_b);
            }

            for il in 0..n_layer {
                ctx0.set_offloading(self.params.should_offload(il));

                let mut current = input_layer.share();

                // self-attention
                {
                    print_shape(&current, "current");

                    let q = ctx0.op_reshape_3d(
                        &ctx0.op_add(
                            &ctx0.op_mul_mat(&self.layers[il].q_w, &current),
                            &self.layers[il].q_b,
                        ),
                        d_head,
                        n_head,
                        input_len,
                    );
                    let q = ctx0.op_permute(&q, (0, 2, 1, 3));
                    print_shape(&q, "q");
                    let q = ctx0.op_cpy(
                        &q,
                        &ctx0.new_tensor_3d(ggml::Type::F32, d_head, input_len, n_head),
                    );

                    let k = ctx0.op_reshape_3d(
                        &ctx0.op_add(
                            &ctx0.op_mul_mat(&self.layers[il].k_w, &current),
                            &self.layers[il].k_b,
                        ),
                        d_head,
                        n_head,
                        input_len,
                    );
                    let k = ctx0.op_permute(&k, (0, 2, 1, 3));
                    print_shape(&k, "k");
                    let k = ctx0.op_cpy(
                        &k,
                        &ctx0.new_tensor_3d(ggml::Type::F16, d_head, input_len, n_head),
                    );

                    let v = ctx0.op_reshape_3d(
                        &ctx0.op_add(
                            &ctx0.op_mul_mat(&self.layers[il].v_w, &current),
                            &self.layers[il].v_b,
                        ),
                        d_head,
                        n_head,
                        input_len,
                    );
                    let mut v = ctx0.op_permute(&v, (0, 2, 1, 3));
                    let mut v = ctx0.op_cpy(
                        &v,
                        &ctx0.new_tensor_3d(ggml::Type::F16, d_head, input_len, n_head),
                    );

                    let mut kq = ctx0.op_mul_mat(&k, &q);

                    // TODO: look into op_scale_inplace and op_soft_max_inplace
                    kq = ctx0.op_scale(
                        &kq,
                        &ctx0.new_f32(1.0 / ((n_embd as f32 / n_head as f32).sqrt())),
                    );
                    kq = ctx0.op_soft_max_inplace(&kq);

                    v = ctx0.op_cont(&ctx0.op_transpose(&v));

                    let kqv = ctx0.op_permute(&ctx0.op_mul_mat(&v, &kq), (0, 2, 1, 3));

                    current = ctx0.op_cpy(
                        &kqv,
                        &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, input_len),
                    );
                }

                // attention output
                current = ctx0.op_add(
                    &ctx0.op_mul_mat(&self.layers[il].o_w, &current),
                    &self.layers[il].o_b,
                );

                // re-add the layer input
                current = ctx0.op_add(&current, &input_layer);

                // attention norm
                {
                    current = ctx0.op_norm(&current);
                    current = ctx0.op_add(
                        &ctx0.op_mul(&current, &self.layers[il].ln_att_w),
                        &self.layers[il].ln_att_b,
                    );
                }

                let att_output = current.share();

                // intermediate output
                current = ctx0.op_mul_mat(&self.layers[il].ff_i_w, &current);
                current = ctx0.op_add(&current, &self.layers[il].ff_i_b);
                current = ctx0.op_gelu(&current);

                // layer output
                current = ctx0.op_mul_mat(&self.layers[il].ff_o_w, &current);
                current = ctx0.op_add(&current, &self.layers[il].ff_o_b);

                // attentions bypass the intermediate layer
                current = ctx0.op_add(&att_output, &current);

                // output norm
                {
                    current = ctx0.op_norm(&current);
                    current = ctx0.op_add(
                        &ctx0.op_mul(&current, &self.layers[il].ln_out_w),
                        &self.layers[il].ln_out_b,
                    );
                }

                // input for next layer
                input_layer = current;
            }
            input_layer = ctx0.op_cont(&ctx0.op_transpose(&input_layer));

            let mut sum = ctx0.new_tensor_2d(llm_base::ElementType::F16, example_len, 1);
            sum = ctx0.set_f32(&sum, 1.0);
            input_layer = ctx0.op_mul_mat(&sum, &input_layer);
            input_layer = ctx0.op_reshape_2d(&input_layer, n_embd,1);
            input_layer = ctx0.op_norm(&input_layer);
            input_layer =
                ctx0.op_scale(&input_layer, &ctx0.new_f32(1.0 / ((n_embd as f32).sqrt())));


            // ctx0.set_offloading(false);
            // pooler
            // let mut sum = ctx0.new_tensor_2d(llm_base::ElementType::F32, input_len, 1);
            // sum = ctx0.set_f32(&sum, 1.0 / (input_len as f32));
            // input_layer = ctx0.op_cpy(
            //     &input_layer,
            //     &ctx0.new_tensor_2d(ggml::Type::F16, input_len, n_embd),
            // );
            // input_layer = ctx0.op_mul_mat(&input_layer, &sum);

            // normalizer
            // let length = ctx0.op_sqrt(&ctx0.op_sum(&ctx0.op_sqr(&input_layer)));

            // input_layer = ctx0.op_scale(&input_layer, &ctx0.op_div(&ctx0.new_f32(1.0), &length));
            // println!("writing dot graph");

            (
                gf,
                GraphOutputs {
                    result: input_layer.share(),
                    embedding_result: input_layer.share(),
                },
            )
        });

        // finish evaluation
        common::read_last_token(session, &outputs.result, n_vocab, input_len);
        common::extract_logits(output_request, &outputs.result, n_vocab, input_len);
        common::extract_embeddings(output_request, &outputs.embedding_result, n_embd, 1);
    }

    /// Run BERT inference for batched examples
    #[tracing::instrument(level = "trace", skip_all)]
    fn batch_evaluate(
        &self,
        session: &mut InferenceSession,
        input_tokens: &[&[TokenId]],
        output_request: &mut OutputRequest,
    ) {
        let batch_size = input_tokens.len();
        // sequence_len is fixed sized (all examples are padded to this length)
        let sequence_len: usize = input_tokens[0].len();

        // For every token, the number of non-padding tokens in the example (used for mean pooling)
        // [sequence_len * batch_size]
        let example_len: Vec<u32> = input_tokens
            .iter()
            .map(|example| {
                let e_lens: Vec<u32> = vec![
                    example
                        .iter()
                        .map(|&i| if i == self.pad_token_id().unwrap() {
                            0
                        } else {
                            1
                        })
                        .sum();
                    sequence_len
                ];
                e_lens
            })
            .flatten()
            .collect();

        // The actual tokens after tokenizer
        // [sequence_len * batch_size]
        let input_tokens = input_tokens
            .iter()
            .flat_map(|row| {
                let mut row = row.to_vec();
                let pad_token_id = self.pad_token_id().unwrap();
                row.resize(sequence_len, pad_token_id);
                row
            })
            .collect::<Vec<_>>();

        // If input token is equal to pad token, then set to the largest negative float32, otherwise make it 0.0
        let attention_mask = input_tokens
            .iter()
            .map(|&tok| {
                if tok == self.pad_token_id().unwrap() {
                    std::f32::MIN
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();

        // 1/(number of non-padding tokens in example) for each token
        let binary_attention_mask = input_tokens
            .iter()
            .zip(example_len.iter())
            .map(|(&tok, &e_len)| {
                if tok == self.pad_token_id().unwrap() {
                    0.0
                } else {
                    1.0 / e_len.to_f32().unwrap()
                }
            })
            .collect::<Vec<_>>();

        // batch_size * sequence_len
        let input_len = input_tokens.len();

        let _session_len = session.n_past;
        let _ctx_size = self.params.context_size;

        let Hyperparameters {
            n_vocab,
            n_max_tokens: _,
            n_embd,
            n_intermediate: _,
            n_head,
            n_layer,
            file_type: _,
        } = self.hyperparameters;

        let d_head = n_embd / n_head;

        let outputs = session.compute(self.context.clone(), &input_tokens, |builder| {
            let mut ctx0 = builder.ctx0.borrow_mut();
            let gf = ggml::ComputationGraph::new();

            let embd = builder.embd;

            let mut input_layer = ctx0.op_get_rows(&self.word_embeddings, embd);

            // Write attention mask to tensor
            let mut attention_mask_tensor =
                ctx0.new_tensor_4d(llm_base::ElementType::F32, input_len, 1, 1, 1);
            unsafe { attention_mask_tensor.write_data(bytemuck::cast_slice(&attention_mask)) };

            let mut bin_attention_mask_tensor =
                ctx0.new_tensor_4d(llm_base::ElementType::F32, input_len, 1, 1, 1);
            unsafe {
                bin_attention_mask_tensor.write_data(bytemuck::cast_slice(&binary_attention_mask))
            };

            // IL = word_embeddings + token_types + position_embeddings
            {
                // token-types: a zero tensor
                let mut token_types = ctx0.new_tensor_1d(llm_base::ElementType::I32, input_len);
                token_types.zero_data();

                // position embeddings: another tensor
                let position_buf: Vec<i32> = (0..input_len as i32)
                    .map(|x| x % sequence_len.to_i32().unwrap())
                    .collect();
                let mut positions = ctx0.new_tensor_1d(llm_base::ElementType::I32, input_len);
                unsafe { positions.write_data(bytemuck::cast_slice(&position_buf)) };

                // IL += token_types
                input_layer = ctx0.op_add(
                    &input_layer,
                    &ctx0.op_get_rows(&self.token_type_embeddings, &token_types),
                );

                // IL += position_embeddings
                input_layer = ctx0.op_add(
                    &input_layer,
                    &ctx0.op_get_rows(&self.position_embeddings, &positions),
                );
            }

            // Reshape input to (n_embed, sequence_len, batch_size)
            input_layer = ctx0.op_reshape_3d(&input_layer, n_embd, sequence_len, batch_size);

            // embd norm
            {
                input_layer = ctx0.op_norm(&input_layer);
                input_layer = ctx0.op_add(&ctx0.op_mul(&input_layer, &self.ln_e_w), &self.ln_e_b);
            }

            for il in 0..n_layer {
                ctx0.set_offloading(self.params.should_offload(il));

                let mut current = input_layer.share();

                // self-attention
                {
                    print_shape(&current, "current");
                    let q_current = ctx0.op_reshape_4d(
                        &ctx0.op_add(
                            &ctx0.op_mul_mat(&self.layers[il].q_w, &current),
                            &self.layers[il].q_b,
                        ),
                        d_head,
                        n_head,
                        sequence_len,
                        batch_size,
                    );
                    let q = ctx0.op_permute(&q_current, (0, 2, 1, 3));

                    print_shape(&q, "q");
                    let q = ctx0.op_cpy(
                        &q,
                        &ctx0.new_tensor_4d(
                            ggml::Type::F32,
                            d_head,
                            sequence_len,
                            n_head,
                            batch_size,
                        ),
                    );

                    let k_current = ctx0.op_reshape_4d(
                        &ctx0.op_add(
                            &ctx0.op_mul_mat(&self.layers[il].k_w, &current),
                            &self.layers[il].k_b,
                        ),
                        d_head,
                        n_head,
                        sequence_len,
                        batch_size,
                    );
                    let k = ctx0.op_permute(&k_current, (0, 2, 1, 3));
                    print_shape(&k, "k");
                    let k = ctx0.op_cpy(
                        &k,
                        &ctx0.new_tensor_4d(
                            ggml::Type::F16,
                            d_head,
                            sequence_len,
                            n_head,
                            batch_size,
                        ),
                    );

                    let v_current = ctx0.op_reshape_4d(
                        &ctx0.op_add(
                            &ctx0.op_mul_mat(&self.layers[il].v_w, &current),
                            &self.layers[il].v_b,
                        ),
                        d_head,
                        n_head,
                        sequence_len,
                        batch_size,
                    );
                    let mut v = ctx0.op_permute(&v_current, (0, 2, 1, 3));
                    print_shape(&v, "v");
                    v = ctx0.op_cpy(
                        &v,
                        &ctx0.new_tensor_4d(
                            ggml::Type::F16,
                            d_head,
                            sequence_len,
                            n_head,
                            batch_size,
                        ),
                    );

                    // Merge batch and heads dimensions
                    let q = ctx0.op_reshape_3d(&q, d_head, sequence_len, n_head * batch_size);
                    let k = ctx0.op_reshape_3d(&k, d_head, sequence_len, n_head * batch_size);
                    v = ctx0.op_reshape_3d(&v, d_head, sequence_len, n_head * batch_size);

                    let mut kq = ctx0.op_mul_mat(&k, &q);

                    kq = ctx0.op_scale(
                        &kq,
                        &ctx0.new_f32(1.0 / ((n_embd as f32 / n_head as f32).sqrt())),
                    );

                    // Split batch and head in separate dimensions
                    let kq =
                        ctx0.op_reshape_4d(&kq, sequence_len, sequence_len, n_head, batch_size);
                    // Move batch dim to the first dim
                    let kq = ctx0.op_permute(&kq, (0, 3, 2, 1));
                    let kq = ctx0.op_cpy(
                        &kq,
                        &ctx0.new_tensor_4d(
                            ggml::Type::F32,
                            sequence_len,
                            batch_size,
                            n_head,
                            sequence_len,
                        ),
                    );
                    // Merge seq_len and batch dimension
                    let kq =
                        ctx0.op_reshape_4d(&kq, sequence_len * batch_size, n_head, sequence_len, 1);

                    // attention mask is 1D [sequence_len*batch_size]
                    // Add is broadcasted
                    let kq = ctx0.op_add(&kq, &attention_mask_tensor);
                    // Reverting dimensions back to [seq_len, seq_len, n_head, bsz]
                    let kq =
                        ctx0.op_reshape_4d(&kq, sequence_len, batch_size, n_head, sequence_len);
                    let kq = ctx0.op_permute(&kq, (0, 3, 2, 1));
                    let kq = ctx0.op_cpy(
                        &kq,
                        &ctx0.new_tensor_4d(
                            ggml::Type::F32,
                            sequence_len,
                            sequence_len,
                            n_head,
                            batch_size,
                        ),
                    );

                    // Merge head and batch dim
                    let kq =
                        ctx0.op_reshape_4d(&kq, sequence_len, sequence_len, n_head * batch_size, 1);
                    let kq = ctx0.op_soft_max(&kq);

                    v = ctx0.op_cont(&ctx0.op_transpose(&v));
                    let kqv = &ctx0.op_mul_mat(&v, &kq);
                    // Reshape back to 4d
                    let kqv = ctx0.op_reshape_4d(kqv, d_head, sequence_len, n_head, batch_size);
                    let kqv = ctx0.op_permute(&kqv, (0, 2, 1, 3));
                    current = ctx0.op_cpy(
                        &kqv,
                        &ctx0.new_tensor_3d(ggml::Type::F32, n_embd, sequence_len, batch_size),
                    );
                }

                // attention output
                current = ctx0.op_add(
                    &ctx0.op_mul_mat(&self.layers[il].o_w, &current),
                    &self.layers[il].o_b,
                );

                // re-add the layer input
                current = ctx0.op_add(&current, &input_layer);

                // attention norm
                {
                    current = ctx0.op_norm(&current);
                    current = ctx0.op_add(
                        &ctx0.op_mul(&current, &self.layers[il].ln_att_w),
                        &self.layers[il].ln_att_b,
                    );
                }

                let att_output = current.share();

                // intermediate output
                current = ctx0.op_mul_mat(&self.layers[il].ff_i_w, &current);
                current = ctx0.op_add(&current, &self.layers[il].ff_i_b);
                current = ctx0.op_gelu(&current);

                // layer output
                current = ctx0.op_mul_mat(&self.layers[il].ff_o_w, &current);
                current = ctx0.op_add(&current, &self.layers[il].ff_o_b);

                // attentions bypass the intermediate layer
                current = ctx0.op_add(&att_output, &current);

                // output norm
                {
                    current = ctx0.op_norm(&current);
                    current = ctx0.op_add(
                        &ctx0.op_mul(&current, &self.layers[il].ln_out_w),
                        &self.layers[il].ln_out_b,
                    );
                }

                // input for next layer
                input_layer = current;
            }

            // Merge seq_len and batch in first dimension
            input_layer = ctx0.op_cont(&ctx0.op_permute(&input_layer, (2, 0, 1, 3)));
            input_layer = ctx0.op_reshape_2d(&input_layer, batch_size * sequence_len, n_embd);
            // Multiplicative masking + weighting
            input_layer = ctx0.op_mul(&input_layer, &bin_attention_mask_tensor);
            input_layer = ctx0.op_reshape_3d(&input_layer, sequence_len, batch_size, n_embd);
            // Reduce_sum over first dimension (pooling)
            let mut sum = ctx0.new_tensor_2d(llm_base::ElementType::F16, sequence_len, 1);
            sum = ctx0.set_f32(&sum, 1.0);
            input_layer = ctx0.op_mul_mat(&sum, &input_layer);
            input_layer = ctx0.op_cont(&ctx0.op_permute(&input_layer, (2, 1, 0, 3)));
            // Normalization
            input_layer = ctx0.op_norm(&input_layer);
            input_layer =
                ctx0.op_scale(&input_layer, &ctx0.new_f32(1.0 / ((n_embd as f32).sqrt())));

            (
                gf,
                GraphOutputs {
                    result: input_layer.share(),
                    embedding_result: input_layer.share(),
                },
            )
        });

        // finish evaluation
        common::extract_embeddings(
            output_request,
            &outputs.embedding_result,
            n_embd,
            batch_size,
        );
    }

    fn hyperparameters(&self) -> &Self::Hyperparameters {
        &self.hyperparameters
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn context_size(&self) -> usize {
        self.params.context_size
    }

    fn pad_token_id(&self) -> Option<TokenId> {
        self.tokenizer.id("[PAD]".as_bytes())
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        self.tokenizer.id("[PAD]".as_bytes())
    }

    fn eot_token_id(&self) -> TokenId {
        self.tokenizer.id("</s>".as_bytes()).unwrap_or(2)
    }

    fn quantize_tensors() -> Vec<Regex> {
        vec![Regex::new(".*weight").unwrap()]
    }

    fn skip_quantize_tensors() -> Vec<Regex> {
        vec![]
    }

    fn supports_rewind(&self) -> bool {
        true
    }
}

impl llm_base::Hyperparameters for Hyperparameters {
    fn read_ggml(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        Ok(Hyperparameters {
            n_vocab: util::read_i32(reader)?.try_into()?,
            n_max_tokens: util::read_i32(reader)?.try_into()?,
            n_embd: util::read_i32(reader)?.try_into()?,
            n_intermediate: util::read_i32(reader)?.try_into()?,
            n_head: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            file_type: util::read_filetype(reader)?,
        })
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_i32(writer, self.n_max_tokens.try_into()?)?;
        util::write_i32(writer, self.n_embd.try_into()?)?;
        util::write_i32(writer, self.n_intermediate.try_into()?)?;
        util::write_i32(writer, self.n_head.try_into()?)?;
        util::write_i32(writer, self.n_layer.try_into()?)?;
        util::write_i32(writer, self.file_type.into())?;
        Ok(())
    }

    fn n_vocabulary(&self) -> usize {
        self.n_vocab
    }

    fn file_type(&self) -> Option<FileType> {
        Some(self.file_type)
    }

    fn file_type_mut(&mut self) -> Option<&mut FileType> {
        Some(&mut self.file_type)
    }
}

struct Layer {
    // normalization
    ln_att_w: ggml::Tensor,
    ln_att_b: ggml::Tensor,

    ln_out_w: ggml::Tensor,
    ln_out_b: ggml::Tensor,

    // attention
    q_w: ggml::Tensor,
    q_b: ggml::Tensor,
    k_w: ggml::Tensor,
    k_b: ggml::Tensor,
    v_w: ggml::Tensor,
    v_b: ggml::Tensor,

    o_w: ggml::Tensor,
    o_b: ggml::Tensor,

    // ff
    ff_i_w: ggml::Tensor,
    ff_i_b: ggml::Tensor,

    ff_o_w: ggml::Tensor,
    ff_o_b: ggml::Tensor,
}

fn print_shape(t: &ggml::Tensor, name: &str) {
    // println!(
    //     "{name} {} [{}] {:?}",
    //     t.get_type(),
    //     t.is_contiguous(),
    //     t.get_ne()
    // );
}
