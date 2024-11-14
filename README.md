# Prediction-of-Cas12a-activity
Prediction of Cas12a trans-cleavage activity 

This project performs data processing, feature engineering, model prediction, and exports multiple analysis results into Excel files for a specific type of crRNA sequence data. Below is an explanation of each part of the code:
1. Reading and Preprocessing Sequence Data
•  It reads the base sequence data from an Excel file and uses a sliding window to extract 24-base subsequences. For each subsequence:
o      It takes the first 20 bases, extracts the reverse complement of the last 4 bases (using the get_reverse_complement() function), and combines them into a new 24-base sequence.
o      The GC content (proportion of G and C bases) is calculated, and the sequence is one-hot encoded.
3. Feature Expansion
•	Polynomial features are generated for the 24-base sequences, including squared terms and interaction terms between features.
•	The expanded feature matrix is standardized using a pre-trained scaler, preparing it as input for the neural network model.
4. Intermediate Result Prediction with a Neural Network
•	An improved neural network, ImprovedNN (IntermediateNN), is constructed to predict intermediate features of the sequence. The input is the expanded and standardized feature matrix, and the output is a 168-dimensional prediction result.
•	The predictions are saved to an Excel file called predictions_for_crRNA_sequences.xlsx.
5. Merging Features and Prediction Results
•	The original features and predicted intermediate features are combined into a new DataFrame, saved to combined_crRNA_features_predictions.xlsx.
6. Activity Prediction Model
•	Another neural network model, ActivityNN, is built and loaded to predict the activity of the crRNA sequences.
•	Using intermediate features and expanded features as input, it generates activity predictions, which are saved to new_predictions.xlsx.
7. Sequence Decoding, Splitting, and Generating Reverse Complement
•	The one-hot encoded feature matrix is decoded back to the original ATGC base sequences.
•	The decoded sequence is split into specific segments: the first 20 bases, the last 4 bases, and the corresponding reverse complement sequence.
•	The reverse complement sequence represents the reverse complement derived from the first 20 bases of the crRNA sequence, used in gene editing-related studies.
8. Merging and Saving Final Results
•	Finally, all processed, predicted, split, and generated sequence results are saved to final_combined_predictions.xlsx, with multiple sheets:
o	  CRRNA_Data_Expanded: Contains the original and expanded features.
o	  Activity_Predictions: Contains activity prediction results.
o	  Merged_Data_with_Sequence: Contains all merged data.
o	  Split_Sequences: Contains segmented base sequences (first 20 bases, reverse complement, and last 4 bases).
The final results file, final_combined_predictions.xlsx, can be used for further researches related to Cas12a-based nucleic acid detection.
