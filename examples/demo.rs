use std::collections::HashMap;
use wm_fuzzy::{Granularity, TNorm, WMModel, WMModelError};

fn main() -> Result<(), WMModelError> {
    println!("==================================================");
    println!("    METABOLIC SYNDROME - (FAKE) FUZZY INFERENCE CLI      ");
    println!("==================================================");

    println!("\nThe weighted average value stands for the chance to have the SM");
    println!("\nATTENTION: Values only for testing!");

    let mut model = WMModel::new();

    // 1. Carrega as regras salvas
    model.load_rules("examples/rules.json")?;
    println!("✓ Loaded {} rules from rules.json\n", model.rules.len());

    // 2. Configura as variáveis de entrada da Síndrome Metabólica
    let features = vec![
        "age",
        "waist_thigh_ratio",
        "waist_hip_ratio",
        "sleep_hours_per_night",
        "physical_activity_categorized",
    ];

    let labels = vec!["low".to_string(), "medium".to_string(), "high".to_string()];

    let limits_map: HashMap<&str, Vec<f64>> = HashMap::from([
        ("age", vec![20.0, 45.0, 70.0]),
        ("waist_thigh_ratio", vec![1.0, 1.4, 1.8]),
        ("waist_hip_ratio", vec![0.70, 0.85, 1.00]),
        ("sleep_hours_per_night", vec![4.0, 7.0, 10.0]),
        ("physical_activity_categorized", vec![0.0, 2.5, 5.0]),
    ]);

    for &feat in &features {
        model
            .granularity
            .insert(feat.to_string(), Granularity::Three);
        model.labels.insert(feat.to_string(), labels.clone());
        model
            .limits
            .insert(feat.to_string(), limits_map[feat].clone());
    }

    model.build_mf_config();

    // 3. Solicita os antecedentes interativamente ao usuário
    let user_antecedents = model.prompt_antecedents();

    // 4. Executa a inferência difusa
    println!("\n--- INFERENCE RESULT ---");
    model.infer(&user_antecedents, TNorm::Product);

    Ok(())
}
