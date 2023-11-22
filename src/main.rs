use itertools::Itertools;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;



fn compute_lengths(unique_translations: &Vec<(String, Vec<String>, String)>) -> (usize, usize) {
    unique_translations.iter().fold((0, 0), |(ref_len, trans_len), translation| {
        let machine_translation = &translation.2;
        let new_trans_len = trans_len + machine_translation.split_whitespace().count();

        let human_translations = &translation.1;
        let new_ref_len = ref_len + human_translations.iter()
            .map(|human_translation| human_translation.split_whitespace().count())
            .sum::<usize>();

        (new_ref_len, new_trans_len)
    })
}

fn compute_brevity_penalty(reference_length: usize, translation_length: usize) -> f64 {
    if translation_length >= reference_length {
        1.0
    } else {
        (1.0 - (reference_length as f64 / translation_length as f64)).exp()
    }
}

fn n_grams_of(s: &str, n: usize) -> Vec<String> {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .windows(n)
        .map(|window| window.join(" "))
        .collect()
}

fn count_ngrams(ngrams: Vec<String>) -> std::collections::HashMap<String, i32> {
    ngrams.into_iter()
        .fold(std::collections::HashMap::new(), |mut acc, ngram| {
            *acc.entry(ngram).or_insert(0) += 1;
            acc
        })
}


fn main() -> io::Result<()> {
    let path = Path::new("./translations_mt.tsv");
    let file = File::open(&path)?;
    let reader = io::BufReader::new(file);

    let translations: Vec<(String, String, String)> = reader
        .lines()
        .filter_map(|line| {
            let line = line.ok()?;
            let columns: Vec<&str> = line.split('\t').collect();
            let columns_length = columns.len();
            Some((
                columns[1].to_string(),
                columns[columns_length - 2].to_string(),
                columns[columns_length - 1].to_string(),
            ))
        })
        .collect();

    let unique_translations: Vec<(String, Vec<String>, String)> = translations
        .into_iter()
        .group_by(|(first, _, _)| first.clone())
        .into_iter()
        .map(|(first, group)| {
            let mut group: Vec<_> = group.collect();
            group.sort_by(|a, b| a.1.cmp(&b.1));
            let middles: Vec<String> = group.iter().map(|(_, middle, _)| middle.clone()).collect();
            let last = group.last().unwrap().2.clone();
            (first, middles, last)
        })
        .collect();

    let precisions: Vec<f64> = (1..=4).map(|n| {
        let mut capped_counts: usize = 0;
        let mut total_counts: usize = 0;
        for (index, translation) in unique_translations.iter().enumerate() {
            let machine_translation = &translation.2;
            let machine_n_grams= n_grams_of(machine_translation, n);                    
            let machine_n_grams_counts = count_ngrams(machine_n_grams); 

            for human_translation in &translation.1 {
                let human_n_grams: Vec<_> = n_grams_of(human_translation, n);                  
                let human_n_grams_length = human_n_grams.len();
                let human_n_grams_counts = count_ngrams(human_n_grams); 

 
                let machine_n_grams_capped: std::collections::HashMap<String, i32> = machine_n_grams_counts.clone()
                        .into_iter()
                        .map(|(ngram, count)| {
                            let capped_count = std::cmp::min(
                                count,
                                *human_n_grams_counts.get(&ngram).unwrap_or(&0),
                            );
                            (ngram, capped_count)
                        })
                        .collect();
                
                capped_counts += machine_n_grams_capped.values().sum::<i32>() as usize;
                total_counts += human_n_grams_length
            }
        }
        capped_counts as f64 / total_counts as f64
    }).collect();

    let (reference_length, translation_length) = compute_lengths(&unique_translations);
    let brevity_penalty = compute_brevity_penalty(reference_length, translation_length);
    let sum_of_log_precisions: f64 = precisions.iter().map(|p| p.ln()).sum();
    let average_log_precision = sum_of_log_precisions / 4.0;
    let bleu_score = brevity_penalty * average_log_precision.exp();
    Ok(())
}
