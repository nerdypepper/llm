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

#[derive(Clone)]
struct BenchResult {
    elapsed: std::time::Duration,
    query_token_count: usize,
}

impl BenchResult {
    /// number of tokens embedded per millisecond
    fn rate(&self) -> f64 {
        (self.query_token_count as f64) / (self.elapsed.as_millis() as f64)
    }
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} tokens, {} ms",
            self.query_token_count,
            self.elapsed.as_millis(),
        )
    }
}

impl std::ops::Add<BenchResult> for BenchResult {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            elapsed: self.elapsed.add(rhs.elapsed),
            query_token_count: self.query_token_count + rhs.query_token_count,
        }
    }
}

fn main() {
    let args = Args::parse();

    let tokenizer_source = args.to_tokenizer_source();
    let model_architecture = args.model_architecture;
    let model_path = args.model_path;
    let corpus = &include_str!("./vicuna-chat.rs")
        .lines()
        .map(|l| &l[..500.min(l.len())])
        .collect::<Vec<_>>();

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
    let mut results = corpus
        .into_iter()
        .map(|l| get_embeddings(model.as_ref(), &inference_parameters, l))
        .collect::<Vec<_>>();
    results.sort_by(|a, b| {
        a.elapsed
            .cmp(&b.elapsed)
            .then(a.query_token_count.cmp(&b.query_token_count))
    });

    let fastest = results.first().unwrap();
    let slowest = results.last().unwrap();

    println!("slowest: {:.04} tok/ms ({})", slowest.rate(), slowest);
    println!("fastest: {:.04} tok/ms ({})", fastest.rate(), fastest);
    println!(
        "average: {:.04} tok/ms, over {} readings",
        results
            .clone()
            .into_iter()
            .reduce(|acc, x| acc + x)
            .unwrap()
            .rate(),
        results.len(),
    );
}

fn get_embeddings(
    model: &dyn llm::Model,
    _inference_parameters: &llm::InferenceParameters,
    query: &str,
) -> BenchResult {
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
    model.evaluate(&mut session, &query_token_ids, &mut output_request);
    let _embeddings = output_request.embeddings.unwrap();
    BenchResult {
        elapsed: s.elapsed(),
        query_token_count: query_token_ids.len(),
    }
}
