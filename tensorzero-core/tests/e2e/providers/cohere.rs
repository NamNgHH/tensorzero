use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("COHERE_API_KEY") {
        Ok(key) => HashMap::from([("cohere_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "cohere".to_string(),
        model_name: "command-r-plus".into(), // choose correct Cohere model name
        model_provider_name: "cohere".into(),
        credentials: credentials.clone(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "cohere-extra-body".to_string(),
        model_name: "command-r-plus".into(),
        model_provider_name: "cohere".into(),
        credentials: credentials.clone(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "cohere-extra-headers".to_string(),
        model_name: "command-r-plus".into(),
        model_provider_name: "cohere".into(),
        credentials: HashMap::new(), // simulate bad/missing key
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "cohere".to_string(),
            model_name: "command-r-plus".into(),
            model_provider_name: "cohere".into(),
            credentials: credentials.clone(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "cohere-strict".to_string(),
            model_name: "command-r-plus".into(), // stricter / premium model
            model_provider_name: "cohere".into(),
            credentials: credentials.clone(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "cohere_json_mode_off".to_string(),
        model_name: "command-r-plus".into(),
        model_provider_name: "cohere".into(),
        credentials: credentials.clone(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "cohere-dynamic".to_string(),
        model_name: "command-r-plus".into(), // Cohere doesn't have many dynamic variants, use same model
        model_provider_name: "cohere".into(),
        credentials,
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "cohere-shorthand".to_string(),
        model_name: "cohere::command-r-plus".into(),
        model_provider_name: "cohere".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: providers.clone(),
        tool_multi_turn_inference: providers.clone(),
        dynamic_tool_use_inference: providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        shorthand_inference: shorthand_providers.clone(),
    }
}
