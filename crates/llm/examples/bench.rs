use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
struct Args {
    model_architecture: llm::ModelArchitecture,
    model_path: PathBuf,
    #[arg(long, short = 'v')]
    pub tokenizer_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    pub tokenizer_repository: Option<String>,
    #[arg(long)]
    pub use_gpu: Option<bool>,
}
impl Args {
    pub fn to_tokenizer_source(&self) -> llm::TokenizerSource {
        match (&self.tokenizer_path, &self.tokenizer_repository) {
            (Some(_), Some(_)) => {
                panic!("Cannot specify both --tokenizer-path and --tokenizer-repository");
            }
            (Some(path), None) => llm::TokenizerSource::HuggingFaceTokenizerFile(path.to_owned()),
            (None, Some(repo)) => llm::TokenizerSource::HuggingFaceRemote(repo.to_owned()),
            (None, None) => llm::TokenizerSource::Embedded,
        }
    }
}

fn main() {
    let args = Args::parse();

    let tokenizer_source = args.to_tokenizer_source();
    let model_architecture = args.model_architecture;
    let model_path = args.model_path;
    let query = &include_str!("./inference.rs")[..400];
    // let query = include_str!("./inference.rs");

    // Load model
    let mut model_params = llm::ModelParameters::default();
    if args.use_gpu.unwrap_or_default() {
        model_params.use_gpu = true;
        dbg!(&model_params.use_gpu);
    }
    let model = llm::load_dynamic(
        Some(model_architecture),
        &model_path,
        tokenizer_source,
        model_params,
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
    });
    let inference_parameters = llm::InferenceParameters::default();

    // Generate embeddings for query and comparands
    let s = std::time::Instant::now();
    let query_embeddings = get_embeddings(model.as_ref(), &inference_parameters, query);
}

fn get_embeddings(
    model: &dyn llm::Model,
    inference_parameters: &llm::InferenceParameters,
    query: &str,
) -> Vec<f32> {
    let s = std::time::Instant::now();
    let session_config = llm::InferenceSessionConfig {
        ..Default::default()
    };
    let mut session = model.start_session(session_config);
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    let vocab = model.tokenizer();
    let beginning_of_sentence = true;
    let query_token_ids = vocab
        .tokenize(query, beginning_of_sentence)
        .unwrap()
        .iter()
        .map(|(_, tok)| *tok)
        .collect::<Vec<_>>();
    println!("input length: {}", query_token_ids.len());
    model.evaluate(&mut session, &query_token_ids, &mut output_request);
    let embeddings = output_request.embeddings.unwrap();
    let t = s.elapsed().as_millis();
    let l = query_token_ids.len();
    let l_over_t = l as f32 / t as f32;
    println!(
        "input len:{}, time elapsed: {}ms, len/ms: {}",
        l, t, l_over_t
    );
    embeddings
}
