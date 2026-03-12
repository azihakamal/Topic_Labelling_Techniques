# Topic_Labelling_Techniques

## Scripts:

1. train_bert_model_improved.py

Script for training a multi-label BERT classification model on newspaper articles. It prepares the dataset, converts article categories into multi-label vectors, and splits the data using iterative stratified sampling to preserve label distributions. The script trains a transformer-based model, tunes optimal prediction thresholds, evaluates performance on validation and test sets, and saves metrics, predictions, and the trained model. It also supports per-label thresholds and optional top-k prediction evaluation for deeper analysis.

2. train_bert_topk_ranker.py

Script for training a single-label BERT classifier with article-level top-k ranking evaluation. Each article category is treated as a single label, but predictions are evaluated using top-k ranking to capture multiple relevant topics per article. The dataset is split by article ID to avoid data leakage, and performance is measured using precision@k, recall@k, and F1@k. The trained model, evaluation metrics, and prediction outputs are saved for later analysis. 

3. predict_folder_to_one_csv.py

Prediction script for running a trained model on multiple CSV files inside a directory. The script loads a fine-tuned model, processes each CSV file containing article texts, predicts topic labels using a sigmoid-based multi-label approach, and merges all predictions into a single output CSV file. It supports batch inference, configurable thresholds, and optional tracking of the original source file for each prediction. 

4. predict_unseen_single_csv_topk.py

Script for generating top-k topic predictions on a single unseen dataset. It loads a trained BERT model and predicts the most relevant labels for each article using softmax probabilities and top-k ranking. The script processes article text from a CSV file, outputs the predicted labels, and saves the results to a new CSV file along with metadata such as the number of predicted labels and the applied ranking method.

5. cleaning_csv_article.py

Script for cleaning and preprocessing raw newspaper article text stored in CSV files. It removes boilerplate elements such as page numbers, continuation markers, and metadata headers, fixes hyphenated words split across lines, normalizes punctuation and Unicode characters, and standardizes whitespace. The cleaned article text is saved into a new column (clean_article) and written to a separate output directory for further processing in topic labeling pipelines. 

6. combine_csv.py

Utility script that merges multiple CSV files from a specified directory into a single combined dataset. Each input file is read and concatenated into one dataframe, while preserving the original filename in an additional column to track the source of each row. The resulting merged dataset is saved as a new CSV file and can be used for further labeling or model evaluation tasks. 

7. article_labeling_new.py

Script for performing unsupervised topic modeling on newspaper articles using BERTopic. It processes cleaned article text, removes noise and stopwords using spaCy and custom filtering, generates embeddings with SentenceTransformers, and clusters articles into topics using UMAP and HDBSCAN. The model generates descriptive topic labels and assigns them to each article. Results are exported as CSV files containing topic IDs and labels, along with a saved BERTopic model for later inference. 

8. bertopic_gwdg_fixed_categories_topk5.py

Script that maps BERTopic-generated topics to a predefined category taxonomy using an LLM through the GWDG API. The script first predicts the top-k BERTopic topics for each article, then uses a language model to map each topic to the closest matching category from a fixed taxonomy. Results are cached to avoid repeated API calls and saved as CSV files containing both the predicted topics and their mapped categories. 

9. gwdg_api_labelling_new.py

Script for labeling newspaper articles using a large language model via the GWDG API. Each article is classified into multiple categories selected from a predefined taxonomy. The script processes CSV files in parallel, sends article text to the API, retrieves the predicted categories, and ensures exactly k labels per article. It includes checkpointing, retry logic, and error handling to make large-scale labeling robust and fault tolerant. 

10. gwdg_gold_label.py

Script for generating gold-standard labels for evaluation datasets using the GWDG API. Similar to the main labeling script, it assigns exactly k categories to each article from the predefined taxonomy. The script supports checkpointing, retry mechanisms, and parallel processing to efficiently label large datasets while ensuring consistent multi-label outputs.

11. pdf_extraction.py

Script for extracting text from newspaper PDFs using Tesseract OCR. The script converts each PDF page into images using pdf2image, performs OCR to detect text blocks, and groups recognized words into structured blocks. The output is saved as a JSON file where each page contains its detected text blocks, enabling further processing such as article extraction and topic labeling. 

12. pdf_extraction_csv.py

Script that converts OCR-generated JSON files into structured article datasets. It reconstructs complete newspaper articles by grouping pages based on heuristics such as continuation markers (e.g., Fortsetzung) and headline detection. Each reconstructed article is saved with its page range and full text, and the results are exported as CSV files for downstream processing and analysis. 

13. tesseract_bbox_visualizer.py

Utility script for visualizing OCR detection results from Tesseract. The script processes a PDF file, detects text blocks and words, and draws colored bounding boxes around them using OpenCV. Each detected block is highlighted with a distinct color and saved as an annotated image, helping inspect OCR quality and document layout segmentation. 

14. visualisation_newspaper.py

Script for comparing topic distributions generated by different classification methods, including BERT, BERTopic, and LLM-based labeling. It loads labeled datasets from multiple CSV files, counts the most frequent topics, and generates bar charts using Seaborn to visualize the top topics for each method. The resulting figures are saved for analysis and reporting in the thesis. 

15. visualize_barchart.py

Script for analyzing and visualizing label distributions across different labeling techniques. It calculates the frequency of labels from supervised classification, zero-shot classification, and hybrid approaches, then generates bar charts showing the top labels for each method. The resulting plots help compare how topic distributions differ across classification techniques. 

16. visualize_bertopic_png.py

Script for generating interactive and static visualizations from a trained BERTopic model. It loads the trained topic model and dataset, then produces visual outputs such as topic bar charts, topic maps, hierarchical topic structures, and similarity heatmaps. The figures are exported as both HTML and PNG files for exploration and inclusion in reports or presentations. 

17. visualize_gwdg.py

Script that aggregates labeled datasets produced by the GWDG LLM labeling pipeline and generates summary visualizations. It combines multiple CSV files into a single dataset, calculates label frequencies, and produces charts such as bar plots to illustrate the distribution of thematic labels across articles while filtering out labeling errors. 

18. train_bert_topk_ranker.py

Script for training a BERT-based classifier with top-k ranking evaluation for article categorization. It trains a transformer model on labeled articles, splits the dataset by article ID to prevent data leakage, and evaluates predictions using top-k metrics such as precision@k, recall@k, and F1@k. The trained model, evaluation metrics, and predicted labels are saved for later analysis and comparison.