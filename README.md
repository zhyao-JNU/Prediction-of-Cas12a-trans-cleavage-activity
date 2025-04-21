# Prediction of Cas12a trans-cleavage activity 

This project performs data processing, feature engineering, model prediction, and exports multiple analysis results into Excel files for a specific type of crRNA sequence data. Below is an explanation of each part of the code (Model application - Prediction of Cas12a trans-cleavage activity for new targets.py):

1. Reverse Complement Function and One-Hot Encoding:
   - A function is defined to compute the reverse complement of a given nucleotide sequence. It converts each base to its complementary base (A ↔ T, G ↔ C), and then reverses the order.
   - A dictionary maps nucleotides ('A', 'T', 'G', 'C') to one-hot encoded vectors (e.g., A → [1, 0, 0, 0], T → [0, 1, 0, 0]). A function is provided to encode each base accordingly.

2. GC Content Calculation:
   - A function computes the GC content (the percentage of G and C bases) for a given sequence. Only the first 24 bases (20 bases + 4 bases) are considered for the calculation.

3. Processing the crRNA Sequences:
   - An Excel file containing crRNA sequences is read. The sequence is extracted from the first row.
   - A sliding window approach is applied to the sequence, extracting all possible 24-base long sequences. For each sequence:
     - The first 20 bases are taken as-is, and the last 4 bases are reversed and complemented.
     - A new sequence is created by combining the original first 20 bases, the reverse complement of the last 4 bases, and the last 4 bases again.
     - GC content for the new sequence is computed, and the sequence is one-hot encoded base-by-base.
   - This data is then saved as a new Excel file with the one-hot encoded sequences and their corresponding GC content.

4. Feature Engineering:
   - The processed data is loaded and a sliding window is applied to generate a 28-base long sequence with one-hot encoding and GC content.
   - Polynomial features are added for the nucleotide bases, including quadratic features and interaction terms between different bases.
   - These new features are standardized using a pre-trained scaler and stored.

5. Model Prediction:
   - A deep neural network model (`IntermediateNN`) is defined with multiple hidden layers and batch normalization.
   - The model is loaded from a saved checkpoint (`improved_nn_model.pth`), and the features are passed through the model to obtain intermediate predictions.
   - The predictions are saved into a new Excel file.

6. Activity Prediction:
   - Another neural network model (`ActivityNN`) is used to predict the biological activity of the crRNA sequences.
   - The model is loaded from a saved checkpoint (`activity_model.pth`), and new features (including the polynomial features and GC content) are passed to predict the activity.
   - The results are saved into another Excel file.

7. Combining Data and Generating Final Output:
   - The final dataset combines the expanded features, the activity predictions, and the original sequences.
   - The one-hot encoded sequences are decoded back into the original base sequences (ATGC format).
   - The sequences are split into "Target Site (TS)", "Guide", and "PAM" components:
     - TS: The first 20 bases.
     - Guide: The reverse complement of the first 20 bases.
     - PAM: The last 4 bases in reverse complement form.
   - The sequences and activity predictions are saved into a new Excel file, containing multiple sheets with the processed data.

8. Saving Final Predictions:
   - All the data, including activity predictions, feature-expanded sequences, and split sequences, are saved into a final Excel file containing multiple sheets for further analysis.

Final Output:
- Predictions for crRNA Sequences: The model's predicted activity is saved.
- Decoded Sequences: The one-hot encoded sequences are decoded back into their original nucleotide format (ATGC).
- Split Sequences: The crRNA sequences are split into TS, guide, and PAM components.

Output Files:
- processed_crRNA_sequences.xlsx: Contains the one-hot encoded sequences and GC content.
- predictions_for_crRNA_sequences.xlsx: Contains the model's intermediate predictions.
- combined_crRNA_features_predictions.xlsx: Combines original features and predictions.
- final_combined_predictions.xlsx: Contains all data, predictions, and sequence splits.

The final results file, final_combined_predictions.xlsx, can be used for further researches related to Cas12a-based nucleic acid detection.


