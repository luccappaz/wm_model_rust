use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use thiserror::Error;

/// Represents the antecedent conjunction (IF feature_1 IS group_1 AND ...).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Antecedents {
    pub data: HashMap<String, String>,
}

impl Antecedents {
    pub fn new(data: HashMap<String, String>) -> Self {
        Self { data }
    }

    /// Computes the Euclidean distance between two antecedent sets in the continuous domain
    /// using the centers ($b$ vertex) of each fuzzy membership function.
    pub fn distance(
        &self,
        other: &Antecedents,
        centers: &HashMap<String, HashMap<String, f64>>,
    ) -> f64 {
        let mut sum_sq = 0.0;
        for (feature, group) in &self.data {
            if let Some(other_group) = other.data.get(feature) {
                let c1 = centers.get(feature).and_then(|m| m.get(group));
                let c2 = centers.get(feature).and_then(|m| m.get(other_group));

                match (c1, c2) {
                    (Some(&c1), Some(&c2)) => sum_sq += (c1 - c2).powi(2),
                    _ => return f64::INFINITY,
                }
            } else {
                return f64::INFINITY;
            }
        }
        sum_sq.sqrt()
    }
}

impl std::fmt::Display for Antecedents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let conditions: Vec<String> = self
            .data
            .iter()
            .map(|(feature, group)| format!("{} is {}", feature, group))
            .collect();
        write!(f, "{}", conditions.join(" AND "))
    }
}

/// Represents an individual fuzzy rule.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FuzzyRule {
    pub antecedents: Antecedents,
    pub weighted_average: f64,
    pub weighted_variance: f64,
    pub doc: f64,
    pub support: f64,
}

impl std::fmt::Display for FuzzyRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IF {} THEN y is centered at {:.4} with variance of {:.4} and doc of {:.4}",
            self.antecedents, self.weighted_average, self.weighted_variance, self.doc
        )
    }
}

/// Information about a rule activated during continuous inference.
pub struct ActivatedRule<'a> {
    pub rule: &'a FuzzyRule,
    pub firing_strength: f64,
}

/// Linguistic partitioning granularity for a feature.
#[derive(Clone, Copy)]
pub enum Granularity {
    Three = 3,
    Five = 5,
}

/// T-norm operator used for rule antecedent aggregation.
#[derive(Clone, Copy)]
pub enum TNorm {
    Product,
    Min,
}

/// Binary classification evaluation metrics.
#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: [[usize; 2]; 2],
}

/// Errors produced by the `WMModel` pipeline.
#[derive(Error, Debug)]
pub enum WMModelError {
    #[error("No rules loaded! Or generate or load new rules")]
    NoRulesLoaded,
    #[error("Granularity is empty or invalid.")]
    InvalidGranularity,
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Main Wang-Mendel model engine.
pub struct WMModel {
    pub granularity: HashMap<String, Granularity>,
    pub labels: HashMap<String, Vec<String>>,
    pub limits: HashMap<String, Vec<f64>>,
    pub mb_centers: HashMap<String, HashMap<String, f64>>,
    pub rules: Vec<FuzzyRule>,
    pub mf_configs: HashMap<String, HashMap<String, [f64; 3]>>,
    pub threshold: f64,
}

impl WMModel {
    /// Creates an empty instance of `WMModel`.
    pub fn new() -> Self {
        Self {
            granularity: HashMap::new(),
            labels: HashMap::new(),
            limits: HashMap::new(),
            mb_centers: HashMap::new(),
            rules: Vec::new(),
            mf_configs: HashMap::new(),
            threshold: 0.5,
        }
    }

    /// Computes the triangular membership degree $\mu(x; a, b, c)$.
    ///
    /// # Parameters
    /// - `x`: Input scalar value.
    /// - `[a, b, c]`: Left vertex ($a$), center peak ($b$), and right vertex ($c$).
    ///
    /// # Examples
    /// ```
    /// use wm_fuzzy::WMModel;
    ///
    /// assert_eq!(WMModel::mb_function(20.0, [10.0, 20.0, 30.0]), 1.0);
    /// assert_eq!(WMModel::mb_function(15.0, [10.0, 20.0, 30.0]), 0.5);
    /// assert_eq!(WMModel::mb_function(35.0, [10.0, 20.0, 30.0]), 0.0);
    /// ```
    fn mb_function(x: f64, [a, b, c]: [f64; 3]) -> f64 {
        if a == b && x <= b {
            return 1.0;
        }
        if b == c && x >= b {
            return 1.0;
        }
        if x == b {
            return 1.0;
        }
        if a != b && x > a && x < b {
            return (x - a) / (b - a);
        }
        if b != c && x > b && x < c {
            return (c - x) / (c - b);
        }
        0.0
    }

    /// Computes the $[a, b, c]$ parameters of a triangular membership function.
    pub fn get_abc(i: usize, granularity: Granularity, limits: &[f64]) -> [f64; 3] {
        let n = granularity as usize;
        let prev = if i == 0 { limits[0] } else { limits[i - 1] };
        let curr = limits[i];
        let next = if i == n - 1 { limits[i] } else { limits[i + 1] };
        [prev, curr, next]
    }

    /// Builds membership function configurations and centers for all configured features.
    pub fn build_mf_config(&mut self) -> &HashMap<String, HashMap<String, [f64; 3]>> {
        let mut mf_configs = HashMap::new();
        let mut mb_centers = HashMap::new();

        for (feature, limits) in &self.limits {
            let labels = &self.labels[feature];
            let granularity = self.granularity[feature];
            let gran_count = granularity as usize;

            let mut feature_mfs = HashMap::new();
            let mut feature_centers = HashMap::new();

            let iter = 0..gran_count;
            for i in iter {
                let group_name = labels[i].clone();
                let abc = Self::get_abc(i, granularity, limits);
                feature_centers.insert(group_name.clone(), abc[1]);
                feature_mfs.insert(group_name, abc);
            }

            mf_configs.insert(feature.clone(), feature_mfs);
            mb_centers.insert(feature.clone(), feature_centers);
        }

        self.mf_configs = mf_configs;
        self.mb_centers = mb_centers;
        &self.mf_configs
    }

    /// Calculates membership degrees $\mu$ for an individual sample across all fuzzy partitions.
    fn get_row_memberships(
        &self,
        row: &HashMap<String, f64>,
    ) -> HashMap<String, HashMap<String, f64>> {
        let mut memberships = HashMap::new();

        for (feature, fuzzy_sets) in &self.mf_configs {
            if let Some(&value) = row.get(feature) {
                let mut group_mus = HashMap::new();
                for (group_name, &abc) in fuzzy_sets {
                    let mu = Self::mb_function(value, abc);
                    group_mus.insert(group_name.clone(), mu);
                }
                memberships.insert(feature.clone(), group_mus);
            }
        }
        memberships
    }

    /// Generates candidate antecedent combinations via Cartesian product.
    fn extract_antecedents(&self, feature_names: &[String]) -> Vec<Antecedents> {
        let mut combinations: Vec<HashMap<String, String>> = vec![HashMap::new()];

        for feature in feature_names {
            let labels = match self.labels.get(feature) {
                Some(l) if !l.is_empty() => l,
                _ => continue,
            };

            let mut next_combinations = Vec::new();
            for current_map in &combinations {
                for label in labels {
                    let mut updated_map = current_map.clone();
                    updated_map.insert(feature.clone(), label.clone());
                    next_combinations.push(updated_map);
                }
            }
            combinations = next_combinations;
        }

        combinations
            .into_iter()
            .map(|data| Antecedents { data })
            .collect()
    }

    /// Extracts and filters fuzzy rules from training data using the Wang-Mendel algorithm.
    pub fn generate_rules(
        &mut self,
        x: &[HashMap<String, f64>],
        y: &[f64],
        feature_names: &[String],
        t_norm: TNorm,
        min_support: f64,
        min_confidence: f64,
    ) -> Vec<FuzzyRule> {
        let n_samples = y.len();
        if n_samples == 0 {
            return Vec::new();
        }

        self.build_mf_config();

        let candidate_antecedents = self.extract_antecedents(feature_names);
        if candidate_antecedents.is_empty() {
            eprintln!("[WARN] No candidate antecedents extracted!");
            return Vec::new();
        }

        // Pré-calcula os memberships de cada linha para não recomputar repetidamente
        let row_memberships: Vec<HashMap<String, HashMap<String, f64>>> =
            x.iter().map(|row| self.get_row_memberships(row)).collect();

        let (y_min, y_max) = y
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });
        let variance_range = (y_max - y_min).max(0.5);

        let mut generated_rules = Vec::new();

        for antecedents in candidate_antecedents {
            // Força de disparo da regra ao longo de todas as amostras usando a t-norm
            let rule_strength: Vec<f64> = row_memberships
                .iter()
                .map(|mems| Self::calculate_firing_strength(&antecedents, mems, t_norm))
                .collect();

            let sum_strength: f64 = rule_strength.iter().sum();

            if sum_strength < 0.0 {
                continue;
            }

            let support = sum_strength / (n_samples as f64);
            if support < min_support {
                continue;
            }

            let weighted_sum: f64 = y.iter().zip(&rule_strength).map(|(&yi, &wi)| yi * wi).sum();
            let weighted_avg = weighted_sum / sum_strength;

            let weighted_var_sum: f64 = y
                .iter()
                .zip(&rule_strength)
                .map(|(&yi, &wi)| (yi - weighted_avg).abs() * wi)
                .sum();
            let weighted_variance = weighted_var_sum / sum_strength;

            let doc = (1.0 - (weighted_variance / variance_range)).clamp(0.0, 1.0);
            if doc < min_confidence {
                continue;
            }

            generated_rules.push(FuzzyRule {
                antecedents,
                support,
                weighted_average: weighted_avg,
                weighted_variance,
                doc,
            });
        }

        self.rules.extend(generated_rules.clone());
        generated_rules
    }

    fn fuzzify_row(&self, row: &HashMap<String, f64>) -> HashMap<String, String> {
        let mut antecedents = HashMap::new();

        for (feature, fuzzy_sets) in &self.mf_configs {
            if let Some(&value) = row.get(feature) {
                let mut best_group = String::new();
                let mut max_mu = -1.0;

                for (group_name, &abc) in fuzzy_sets {
                    let mu = Self::mb_function(value, abc);
                    if mu > max_mu {
                        max_mu = mu;
                        best_group = group_name.clone();
                    }
                }
                antecedents.insert(feature.clone(), best_group);
            }
        }
        antecedents
    }

    fn min_distance_rule(&self, antecedents_dict: HashMap<String, String>) -> Option<&FuzzyRule> {
        if self.rules.is_empty() {
            return None;
        }

        let target_antecedents = Antecedents::new(antecedents_dict);
        let mut nearest_rule: Option<(&FuzzyRule, f64)> = None;

        for rule in &self.rules {
            let dist = target_antecedents.distance(&rule.antecedents, &self.mb_centers);
            if dist.is_finite() {
                match nearest_rule {
                    Some((_, min_dist)) if dist < min_dist => {
                        nearest_rule = Some((rule, dist));
                    }
                    None => {
                        nearest_rule = Some((rule, dist));
                    }
                    _ => {}
                }
            }
        }

        nearest_rule.map(|(rule, _)| rule)
    }

    fn calculate_firing_strength(
        antecedents: &Antecedents,
        memberships: &HashMap<String, HashMap<String, f64>>,
        t_norm: TNorm,
    ) -> f64 {
        match t_norm {
            TNorm::Product => {
                let mut strength = 1.0;
                for (feature, group) in &antecedents.data {
                    let mu = memberships
                        .get(feature)
                        .and_then(|groups| groups.get(group))
                        .copied()
                        .unwrap_or(0.0);
                    strength *= mu;
                    if strength == 0.0 {
                        break;
                    }
                }
                strength
            }
            TNorm::Min => {
                let mut min_mu = 1.0;
                for (feature, group) in &antecedents.data {
                    let mu = memberships
                        .get(feature)
                        .and_then(|groups| groups.get(group))
                        .copied()
                        .unwrap_or(0.0);
                    if mu < min_mu {
                        min_mu = mu;
                    }
                    if min_mu == 0.0 {
                        break;
                    }
                }
                min_mu
            }
        }
    }

    /// Performs continuous and binary prediction on a test dataset.
    pub fn predict(
        &mut self,
        x_test: &[HashMap<String, f64>],
        threshold: Option<f64>,
        fallback: f64,
        t_norm: TNorm,
    ) -> Result<(Vec<f64>, Vec<u32>), WMModelError> {
        let thresh = threshold.unwrap_or(self.threshold);
        if self.rules.is_empty() {
            return Err(WMModelError::NoRulesLoaded);
        }

        self.build_mf_config();

        let mut continuous_preds = Vec::with_capacity(x_test.len());
        let mut binary_preds = Vec::with_capacity(x_test.len());

        for row in x_test {
            let memberships = self.get_row_memberships(row);
            let mut total_firing_strength = 0.0;
            let mut weighted_output_sum = 0.0;

            for rule in &self.rules {
                let firing_strength =
                    Self::calculate_firing_strength(&rule.antecedents, &memberships, t_norm);

                if firing_strength > 0.0 {
                    weighted_output_sum += firing_strength * rule.weighted_average;
                    total_firing_strength += firing_strength;
                }
            }

            let y_pred = if total_firing_strength > 0.0 {
                weighted_output_sum / total_firing_strength
            } else {
                let ant_dict = self.fuzzify_row(row);
                if let Some(nearest) = self.min_distance_rule(ant_dict) {
                    nearest.weighted_average
                } else {
                    fallback
                }
            };

            continuous_preds.push(y_pred);
            binary_preds.push(if y_pred >= thresh { 1 } else { 0 });
        }

        Ok((continuous_preds, binary_preds))
    }

    /// Evaluates model performance on test data and serializes the metrics to `fuzzy_results.json`
    /// in the current working directory.
    pub fn evaluate(
        &mut self,
        x_test: &[HashMap<String, f64>],
        y_test: &[u32],
        fallback: Option<f64>,
    ) -> Result<EvaluationMetrics, WMModelError> {
        let (_, y_pred_bin) =
            self.predict(x_test, None, fallback.unwrap_or(0.5), TNorm::Product)?;

        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_val = 0;

        for (&yt, &yp) in y_test.iter().zip(y_pred_bin.iter()) {
            match (yt, yp) {
                (1, 1) => tp += 1,
                (0, 0) => tn += 1,
                (0, 1) => fp += 1,
                (1, 0) => fn_val += 1,
                _ => {}
            }
        }

        let total = (tp + tn + fp + fn_val) as f64;
        let accuracy = if total > 0.0 {
            (tp + tn) as f64 / total
        } else {
            0.0
        };
        let precision = if (tp + fp) > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if (tp + fn_val) > 0 {
            tp as f64 / (tp + fn_val) as f64
        } else {
            0.0
        };
        let f1_score = if (precision + recall) > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        let metrics = EvaluationMetrics {
            accuracy,
            precision,
            recall,
            f1_score,
            confusion_matrix: [[tn, fp], [fn_val, tp]],
        };

        let output_path = std::env::current_dir()?.join("fuzzy_results.json");
        let file = File::create(&output_path)?;
        serde_json::to_writer_pretty(file, &metrics)?;

        println!("✅ Metrics saved in: {:?}", output_path);

        Ok(metrics)
    }

    /// Serializes and saves the rule base to a JSON file.
    pub fn save_rules<P: AsRef<Path>>(&self, path: P) -> Result<(), WMModelError> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &self.rules)?;
        Ok(())
    }

    /// Loads fuzzy rules from a JSON file into `self.rules`.
    pub fn load_rules<P: AsRef<Path>>(&mut self, path: P) -> Result<(), WMModelError> {
        let file = File::open(path)?;
        let loaded_rules: Vec<FuzzyRule> = serde_json::from_reader(file)?;
        self.rules.extend(loaded_rules);
        Ok(())
    }

    /// Finds and prints all rules activated by the provided linguistic antecedents.
    pub fn infer(&self, antecedents: &Antecedents, t_norm: TNorm) -> Vec<ActivatedRule<'_>> {
        let mut crisp_row = HashMap::new();
        for (feature, group) in &antecedents.data {
            if let Some(groups) = self.mf_configs.get(feature) {
                if let Some(abc) = groups.get(group) {
                    crisp_row.insert(feature.clone(), abc[1]);
                }
            } else if let Some(center) = self.mb_centers.get(feature).and_then(|m| m.get(group)) {
                crisp_row.insert(feature.clone(), *center);
            }
        }

        let memberships = self.get_row_memberships(&crisp_row);
        let mut activated = Vec::new();
        let mut total_firing_strength = 0.0;
        let mut weighted_output_sum = 0.0;

        for rule in &self.rules {
            let firing_strength =
                Self::calculate_firing_strength(&rule.antecedents, &memberships, t_norm);

            if firing_strength > 0.0 {
                weighted_output_sum += firing_strength * rule.weighted_average;
                total_firing_strength += firing_strength;

                activated.push(ActivatedRule {
                    rule,
                    firing_strength,
                });
            }
        }

        activated.sort_by(|a, b| {
            b.firing_strength
                .partial_cmp(&a.firing_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if activated.is_empty() {
            println!("\n⚠️ None rules were actiavted for the given antecedents:");
            println!("  {}", antecedents);
        } else {
            println!("\n🔥 Activated Rules (Total: {}):", activated.len());
            for (idx, item) in activated.iter().enumerate() {
                println!(
                    "  [{}] Strength: {:>6.4} | {}",
                    idx + 1,
                    item.firing_strength,
                    item.rule
                );
            }

            if total_firing_strength > 0.0 {
                let y_pred = weighted_output_sum / total_firing_strength;
                println!("\n📊 [Defuzzification / Weighted Mean]:");
                println!("  ↳ y infered: {:.4}", y_pred);
                println!("  ↳ Total strength combined: {:.4}", total_firing_strength);
            }
        }

        activated
    }

    /// Prompts the user interactively in the terminal to select linguistic terms
    /// for each feature, displaying extreme points and centers $[a, b, c]$.
    pub fn prompt_antecedents(&mut self) -> Antecedents {
        if self.mf_configs.is_empty() {
            self.build_mf_config();
        }

        let mut chosen_data = HashMap::new();

        println!("\n==================================================");
        println!("         FUZZY ANTECEDENTS SELECTION            ");
        println!("==================================================");

        for (feature, groups) in &self.mf_configs {
            println!("\n📌 Attributes: \"{}\"", feature);
            println!("Choose one from the semantic values:");

            // Ordena as opções pelo centro 'b' (ordem crescente de grandeza)
            let mut options: Vec<(&String, &[f64; 3])> = groups.iter().collect();
            options.sort_by(|a, b| {
                a.1[1]
                    .partial_cmp(&b.1[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let iter = options.iter().enumerate();
            for (idx, (group_name, abc)) in iter {
                let [a, b, c] = **abc;
                println!(
                    "  [{}] {:<12} -> [a: {:>8.3}, b (center): {:>8.3}, c: {:>8.3}]",
                    idx + 1,
                    group_name,
                    a,
                    b,
                    c
                );
            }

            loop {
                print!(
                    "👉 Type the number (1-{}) or the option name: ",
                    options.len()
                );
                io::stdout().flush().unwrap();

                let mut input = String::new();
                if io::stdin().read_line(&mut input).is_err() {
                    println!("Error to read input. Try again.");
                    continue;
                }

                let input = input.trim();

                if let Ok(num) = input.trim().parse::<usize>()
                    && (1..=options.len()).contains(&num)
                {
                    let (selected_group, _) = options[num - 1];
                    chosen_data.insert(feature.clone(), selected_group.to_string());
                    break;
                }

                if let Some((selected_group, _)) = options
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(input))
                {
                    chosen_data.insert(feature.clone(), selected_group.to_string());
                    break;
                }

                println!(
                    "❌ Invalid option \"{}\"! Choose a value between 1 and {} or the exact name.",
                    input,
                    options.len()
                );
            }
        }

        Antecedents::new(chosen_data)
    }
}

impl Default for WMModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const EPSILON: f64 = 1e-6;

    /// Helper to initialize a fully configured model with two features: "temp" and "pressure"
    fn setup_configured_model() -> WMModel {
        let mut model = WMModel::new();

        // 1. Configure granularities (3 partitions for each feature)
        model
            .granularity
            .insert("temp".to_string(), Granularity::Three);
        model
            .granularity
            .insert("pressure".to_string(), Granularity::Three);

        // 2. Assign linguistic terms
        model.labels.insert(
            "temp".to_string(),
            vec!["low".to_string(), "med".to_string(), "high".to_string()],
        );
        model.labels.insert(
            "pressure".to_string(),
            vec!["low".to_string(), "med".to_string(), "high".to_string()],
        );

        // 3. Define numerical boundaries for the universe of discourse
        model
            .limits
            .insert("temp".to_string(), vec![10.0, 20.0, 30.0]);
        model
            .limits
            .insert("pressure".to_string(), vec![100.0, 200.0, 300.0]);

        model.build_mf_config();
        model
    }

    // ========================================================
    // 1. MATHEMATICAL TESTS & MEMBERSHIP FUNCTIONS (MFs)
    // ========================================================

    #[test]
    fn test_mb_function_triangular_points() {
        let abc = [10.0, 20.0, 30.0];

        // Out of domain boundaries
        assert_eq!(WMModel::mb_function(5.0, abc), 0.0);
        assert_eq!(WMModel::mb_function(35.0, abc), 0.0);

        // Exact vertices
        assert_eq!(WMModel::mb_function(10.0, abc), 0.0);
        assert_eq!(WMModel::mb_function(20.0, abc), 1.0);
        assert_eq!(WMModel::mb_function(30.0, abc), 0.0);

        // Ascending and descending slopes
        assert!((WMModel::mb_function(15.0, abc) - 0.5).abs() < EPSILON);
        assert!((WMModel::mb_function(25.0, abc) - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_mb_function_boundary_shoulders() {
        // Left shoulder function where a == b
        let left_shoulder = [10.0, 10.0, 20.0];
        assert_eq!(WMModel::mb_function(10.0, left_shoulder), 1.0);
        assert!((WMModel::mb_function(15.0, left_shoulder) - 0.5).abs() < EPSILON);

        // Right shoulder function where b == c
        let right_shoulder = [20.0, 30.0, 30.0];
        assert_eq!(WMModel::mb_function(30.0, right_shoulder), 1.0);
        assert!((WMModel::mb_function(25.0, right_shoulder) - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_get_abc_generation() {
        let limits = vec![0.0, 50.0, 100.0];
        let gran = Granularity::Three;

        // First fuzzy set: [0.0, 0.0, 50.0]
        assert_eq!(WMModel::get_abc(0, gran, &limits), [0.0, 0.0, 50.0]);
        // Intermediate fuzzy set: [0.0, 50.0, 100.0]
        assert_eq!(WMModel::get_abc(1, gran, &limits), [0.0, 50.0, 100.0]);
        // Last fuzzy set: [50.0, 100.0, 100.0]
        assert_eq!(WMModel::get_abc(2, gran, &limits), [50.0, 100.0, 100.0]);
    }

    // ========================================================
    // 2. ANTECEDENTS & FUZZY ALGEBRA (T-NORMS)
    // ========================================================

    #[test]
    fn test_extract_antecedents_cartesian_product() {
        let model = setup_configured_model();
        let features = vec!["temp".to_string(), "pressure".to_string()];

        let combinations = model.extract_antecedents(&features);

        // 3 partitions (temp) x 3 partitions (pressure) = 9 total combinations
        assert_eq!(combinations.len(), 9);

        // Verify that every combination includes both variables
        for ant in &combinations {
            assert!(ant.data.contains_key("temp"));
            assert!(ant.data.contains_key("pressure"));
        }
    }

    #[test]
    fn test_antecedents_euclidean_distance() {
        let mut d1 = HashMap::new();
        d1.insert("x".to_string(), "low".to_string());
        d1.insert("y".to_string(), "low".to_string());
        let a1 = Antecedents::new(d1);

        let mut d2 = HashMap::new();
        d2.insert("x".to_string(), "high".to_string());
        d2.insert("y".to_string(), "high".to_string());
        let a2 = Antecedents::new(d2);

        let mut centers = HashMap::new();
        let mut x_centers = HashMap::new();
        x_centers.insert("low".to_string(), 0.0);
        x_centers.insert("high".to_string(), 3.0);
        centers.insert("x".to_string(), x_centers);

        let mut y_centers = HashMap::new();
        y_centers.insert("low".to_string(), 0.0);
        y_centers.insert("high".to_string(), 4.0);
        centers.insert("y".to_string(), y_centers);

        // Euclidean distance in R^2: sqrt((3.0 - 0.0)^2 + (4.0 - 0.0)^2) = sqrt(25) = 5.0
        let dist = a1.distance(&a2, &centers);
        assert!((dist - 5.0).abs() < EPSILON);

        // Missing features must return infinity
        let empty_ant = Antecedents::new(HashMap::new());
        assert_eq!(a1.distance(&empty_ant, &centers), f64::INFINITY);
    }

    #[test]
    fn test_firing_strength_tnorms() {
        let mut ant_data = HashMap::new();
        ant_data.insert("temp".to_string(), "high".to_string());
        ant_data.insert("pressure".to_string(), "low".to_string());
        let rule = FuzzyRule {
            antecedents: Antecedents::new(ant_data),
            support: 0.0,
            weighted_average: 1.0,
            weighted_variance: 0.0,
            doc: 1.0,
        };

        let mut memberships = HashMap::new();
        let mut m_temp = HashMap::new();
        m_temp.insert("high".to_string(), 0.8);
        memberships.insert("temp".to_string(), m_temp);

        let mut m_press = HashMap::new();
        m_press.insert("low".to_string(), 0.5);
        memberships.insert("pressure".to_string(), m_press);

        // TNorm::Product = 0.8 * 0.5 = 0.40
        let strength_prod =
            WMModel::calculate_firing_strength(&rule.antecedents, &memberships, TNorm::Product);
        assert!((strength_prod - 0.40).abs() < EPSILON);

        // TNorm::Min = min(0.8, 0.5) = 0.50
        let strength_min =
            WMModel::calculate_firing_strength(&rule.antecedents, &memberships, TNorm::Min);
        assert!((strength_min - 0.50).abs() < EPSILON);
    }

    // ========================================================
    // 3. TRAINING PIPELINE & INFERENCE
    // ========================================================

    #[test]
    fn test_generate_rules_and_predict_flow() {
        let mut model = setup_configured_model();

        // Synthetic 3-sample calibration dataset
        let mut row1 = HashMap::new();
        row1.insert("temp".to_string(), 10.0); // 100% 'low'
        row1.insert("pressure".to_string(), 100.0); // 100% 'low'

        let mut row2 = HashMap::new();
        row2.insert("temp".to_string(), 20.0); // 100% 'med'
        row2.insert("pressure".to_string(), 200.0); // 100% 'med'

        let mut row3 = HashMap::new();
        row3.insert("temp".to_string(), 30.0); // 100% 'high'
        row3.insert("pressure".to_string(), 300.0); // 100% 'high'

        let x = vec![row1.clone(), row2.clone(), row3.clone()];
        let y = vec![0.1, 0.5, 0.9];
        let features = vec!["temp".to_string(), "pressure".to_string()];

        // Rule extraction using permissive filters
        let generated_rules = model.generate_rules(&x, &y, &features, TNorm::Product, 0.1, 0.0);

        assert!(!generated_rules.is_empty());
        assert_eq!(model.rules.len(), generated_rules.len());

        // Prediction at calibration points
        let (preds_cont, preds_bin) = model
            .predict(&x, Some(0.5), 0.0, TNorm::Product)
            .expect("Prediction failed");

        assert_eq!(preds_cont.len(), 3);
        assert!((preds_cont[0] - 0.1).abs() < EPSILON);
        assert!((preds_cont[1] - 0.5).abs() < EPSILON);
        assert!((preds_cont[2] - 0.9).abs() < EPSILON);

        // Binary classifications with threshold = 0.5
        assert_eq!(preds_bin, vec![0, 1, 1]);
    }

    #[test]
    fn test_predict_fallback_mechanism() {
        let mut model = setup_configured_model();

        // Manually inject a single rule
        let mut ant_data = HashMap::new();
        ant_data.insert("temp".to_string(), "low".to_string());
        ant_data.insert("pressure".to_string(), "low".to_string());

        model.rules.push(FuzzyRule {
            antecedents: Antecedents::new(ant_data),
            support: 0.5,
            weighted_average: 0.15,
            weighted_variance: 0.01,
            doc: 0.95,
        });

        // Out-of-distribution sample (0% membership in known partitions)
        let mut unknown_row = HashMap::new();
        unknown_row.insert("temp".to_string(), 1000.0);
        unknown_row.insert("pressure".to_string(), 1000.0);

        // Prediction triggers nearest-rule fallback
        let (preds, _) = model
            .predict(&[unknown_row], None, 0.999, TNorm::Product)
            .expect("Prediction failed");

        assert_eq!(preds.len(), 1);
        // Falls back to closest rule consequent (0.15) via min_distance_rule
        assert!((preds[0] - 0.15).abs() < EPSILON);
    }

    // ========================================================
    // 4. METRIC EVALUATION & PERSISTENCE
    // ========================================================

    #[test]
    fn test_evaluate_metrics_calculation() {
        let mut model = setup_configured_model();

        // Inject rule for inference
        let mut ant = HashMap::new();
        ant.insert("temp".to_string(), "high".to_string());
        ant.insert("pressure".to_string(), "high".to_string());
        model.rules.push(FuzzyRule {
            antecedents: Antecedents::new(ant),
            support: 1.0,
            weighted_average: 0.8, // Binary output = 1 (threshold 0.5)
            weighted_variance: 0.0,
            doc: 1.0,
        });

        let mut row = HashMap::new();
        row.insert("temp".to_string(), 30.0);
        row.insert("pressure".to_string(), 300.0);

        // 2 samples: one where y_true = 1 (TP) and one where y_true = 0 (FP)
        let x_test = vec![row.clone(), row.clone()];
        let y_test = vec![1, 0];

        let metrics = model
            .evaluate(&x_test, &y_test, None)
            .expect("Evaluation failed");

        // Total = 2 | TP = 1, FP = 1, TN = 0, FN = 0
        assert_eq!(metrics.confusion_matrix, [[0, 1], [0, 1]]);
        assert!((metrics.accuracy - 0.5).abs() < EPSILON);
        assert!((metrics.precision - 0.5).abs() < EPSILON);
        assert!((metrics.recall - 1.0).abs() < EPSILON);

        // Cleanup generated evaluation file
        let _ = fs::remove_file("fuzzy_results.json");
    }

    #[test]
    fn test_save_and_load_rules_roundtrip() {
        let mut model = WMModel::new();

        let mut ant = HashMap::new();
        ant.insert("temp".to_string(), "low".to_string());
        let original_rule = FuzzyRule {
            antecedents: Antecedents::new(ant),
            support: 0.42,
            weighted_average: 0.314,
            weighted_variance: 0.05,
            doc: 0.88,
        };
        model.rules.push(original_rule.clone());

        let temp_path = std::env::temp_dir().join("test_rules_roundtrip.json");

        // Save and reload into a fresh model instance
        model.save_rules(&temp_path).expect("Failed to save rules");

        let mut new_model = WMModel::new();
        new_model
            .load_rules(&temp_path)
            .expect("Failed to load rules");

        assert_eq!(new_model.rules.len(), 1);
        assert_eq!(new_model.rules[0], original_rule);

        // Cleanup temporary file
        let _ = fs::remove_file(temp_path);
    }
}
