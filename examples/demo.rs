use std::collections::HashMap;
use wm_fuzzy::{Granularity, TNorm, WMModel, WMModelError};

fn main() -> Result<(), WMModelError> {
    println!("=== Wang-Mendel Fuzzy Inference System ===");

    let mut model = WMModel::new();

    // 1. Configure universes of discourse and linguistic partitions
    model
        .granularity
        .insert("temperature".to_string(), Granularity::Three);
    model
        .granularity
        .insert("pressure".to_string(), Granularity::Three);

    model.labels.insert(
        "temperature".to_string(),
        vec!["low".to_string(), "medium".to_string(), "high".to_string()],
    );
    model.labels.insert(
        "pressure".to_string(),
        vec!["low".to_string(), "medium".to_string(), "high".to_string()],
    );

    model
        .limits
        .insert("temperature".to_string(), vec![10.0, 25.0, 40.0]);
    model
        .limits
        .insert("pressure".to_string(), vec![80.0, 100.0, 120.0]);

    // 2. Synthetic dataset for rule generation
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();

    let samples = [
        (12.0, 85.0, 0.1),
        (24.0, 100.0, 0.5),
        (38.0, 118.0, 0.9),
        (15.0, 90.0, 0.2),
        (35.0, 110.0, 0.8),
    ];

    for &(temp, pressure, target) in &samples {
        let mut row = HashMap::new();
        row.insert("temperature".to_string(), temp);
        row.insert("pressure".to_string(), pressure);
        x_train.push(row);
        y_train.push(target);
    }

    let features = vec!["temperature".to_string(), "pressure".to_string()];

    // 3. Rule generation and persistence
    println!("\n[1/3] Generating fuzzy rules...");
    let rules = model.generate_rules(&x_train, &y_train, &features, TNorm::Product, 0.01, 0.1);
    println!("Total rules extracted: {}", rules.len());

    model.save_rules("fuzzy_rules.json")?;
    println!("Rules saved to: fuzzy_rules.json");

    // 4. Interactive user inference
    println!("\n[2/3] Interactive Inference Mode");
    let user_antecedents = model.prompt_antecedents();
    model.infer(&user_antecedents, TNorm::Product);

    // 5. Model evaluation
    println!("\n[3/3] Evaluating model metrics...");
    let y_test_bin = vec![0, 1, 1, 0, 1];
    let metrics = model.evaluate(&x_train, &y_test_bin, None)?;

    println!("Accuracy:  {:.2}%", metrics.accuracy * 100.0);
    println!("Precision: {:.4}", metrics.precision);
    println!("Recall:    {:.4}", metrics.recall);
    println!("F1-Score:  {:.4}", metrics.f1_score);

    Ok(())
}
