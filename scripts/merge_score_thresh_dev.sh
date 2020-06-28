python3 merge_score_thresh.py \
    ../data/dev_x.csv \
    ../outputs/cond_prob_info_nous_latlon2d-dev.csv \
    ../outputs/cond_prob_info_nous_latlon2d-dev-scores.csv \
    ../outputs/lang_embedding.csv \
    ../outputs/lang_embedding_probs.csv \
    0.5 0.65 \
    > ../outputs/merge-dev.csv

