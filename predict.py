import pandas as pd
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from typing import List, Tuple
from utils import (
    get_device,
    clean_text,
    contains_phone_number,
    contains_keyword_with_numbers,
    contains_specified_keywords,
    KEYWORD_RULES,
)


# --- Configuration ---
# Old: MODEL_NAME = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
MODEL_NAME = "microsoft/deberta-v3-large"
# NOTE: DeBERTa V3 requires the sentencepiece
# library.
# Install it with: pip install sentencepiece

SPAM_THRESHOLD = 0.5  # Threshold for classifying as spam


class HybridSpamDetector:
    """
    A robust spam detector that combines a transformer model with a weighted rule-based system.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.detector = None
        self.device = get_device()
        self._load_model()

    def _load_model(self):
        """Loads the text-classification pipeline with proper device support."""
        try:
            print(f"🔄 Loading model: {self.model_name}")
            print(f"📱 Using device: {self.device.upper()}")

            # 💡 DeBERTa V3 requires AutoTokenizer/AutoModel and specific device handling

            # 1. Load the Tokenizer (handles the specific DeBERTa V3 tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # 2. Load the Model (handles the large model configuration)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # 3. Handle device placement
            if self.device == "mps":
                # For MPS, move model to device. Pipeline device setting can be tricky.
                model.to(self.device, dtype=torch.float16)
                device_index = self.device
            elif self.device == "cuda":
                # For CUDA, use device index
                device_index = 0
            else:
                # For CPU
                device_index = -1

            # 4. Create the pipeline
            self.detector = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device_index,
                # DeBERTa V3 Large handles up to 512 tokens, truncation is safe.
            )

            print(
                f"✅ Successfully loaded model: {self.model_name} on device {self.device.upper()}"
            )

        except Exception as e:
            print(f"❌ Failed to load model '{self.model_name}'. Error: {e}")
            print("❌ This model is large. Please ensure you have sufficient RAM/VRAM.")
            print("🔄 Attempting fallback to CPU (may be very slow)...")
            # Fallback to CPU-only pipeline logic (no dtype optimization)
            try:
                self.detector = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=-1,  # Force CPU
                )
                print("✅ Model loaded on CPU as fallback")
            except Exception:
                print(
                    "❌ Complete failure to load model. Will use rule-based detection only. Error: {e2}"
                )
                self.detector = None

    def _apply_rules(self, text: str) -> float:
        """
        Calculates a score based on keyword and regex matches.
        The score is capped at 1.0.
        """
        score = 0.0
        text_lower = text.lower()

        # Apply keyword rules
        for keyword, weight in KEYWORD_RULES.items():
            if keyword in text_lower:
                score += weight

        # Apply phone number detection with higher weight
        if contains_phone_number(text):
            score += 0.4

        # Apply keyword + number patterns
        if contains_keyword_with_numbers(text):
            score += 0.35

        return min(score, 1.0)  # Cap the max score to 1.0

    def _has_required_patterns(self, text: str) -> bool:
        """
        Check if text contains required patterns (phone numbers OR specified
        keywords). This is the gatekeeper - only texts with these patterns can
        be classified as spam.
        """
        return (
            contains_phone_number(text)
            or contains_keyword_with_numbers(text)
            or contains_specified_keywords(text)
        )

    def predict_batch(self, texts: List[str]) -> List[Tuple[int, float]]:
        """
        Analyzes a batch of texts for spam using a hybrid model-rule approach.
        Only texts containing required patterns are sent to the LLM for analysis.

        Returns:
            A list of tuples, where each tuple contains (is_spam_flag, confidence_score).
        """
        if not texts:
            return []

        # --- Step 1: Pre-filter texts to find those that need LLM analysis ---
        texts_for_llm = []
        indices_for_llm = []
        for i, text in enumerate(texts):
            if self._has_required_patterns(text):
                texts_for_llm.append(text)
                indices_for_llm.append(i)

        # --- Step 2: Get predictions from the model ONLY for the filtered texts ---
        model_probs = {}  # Use a dict to map original index to probability
        if self.detector and texts_for_llm:
            try:
                batch_size = 32 if self.device == "mps" else 64
                results = self.detector(
                    texts_for_llm,  # <-- Pass only the filtered list to the model
                    truncation=True,
                    padding=True,
                    max_length=512,
                    batch_size=batch_size,
                )

                for i, result in enumerate(results):
                    label = result["label"].upper()
                    score = result["score"]
                    original_index = indices_for_llm[i]  # Map result back to original index

                    if label in ["SPAM", "LABEL_1", "1"]:
                        model_probs[original_index] = score
                    else:
                        model_probs[original_index] = 1.0 - score

            except Exception as e:
                print(
                    f"⚠️ Error during model prediction batch: {e}. Falling back to rules for this batch."
                )
                # If the model fails, model_probs will be empty, and the logic below will rely solely on rules.

        # --- Step 3: Calculate final scores for the entire original batch ---
        final_results = []
        for i, text in enumerate(texts):
            # Check if the current text was one of the ones that had patterns
            if i in indices_for_llm:
                rule_score = self._apply_rules(text)
                # Get model probability; default to 0.0 if LLM failed or had an issue
                model_prob = model_probs.get(i, 0.0)

                # Hybrid Score Calculation
                combined_score = model_prob + (1.0 - model_prob) * rule_score
                combined_score = min(combined_score, 1.0)

                is_spam = 1 if combined_score >= SPAM_THRESHOLD else 0
                confidence = combined_score
            else:
                # If no required patterns were found, it cannot be spam.
                is_spam = 0
                confidence = 0.0

            final_results.append((is_spam, confidence))

        return final_results


def main():
    """Main function to load data, run detection, and save results."""
    try:
        print("Loading data from 'summaries.csv'...")
        df = pd.read_csv("summaries.csv", encoding="utf-8")
        print(f"Loaded {len(df)} posts.")
    except FileNotFoundError:
        print(
            "Error: 'summaries.csv' not found. Please ensure the file is in the same directory."
        )
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # --- Data Cleaning ---
    print("Cleaning and preparing text...")
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_summary"] = df["summary"].apply(clean_text)
    df["combined_text"] = df["clean_title"] + " " + df["clean_summary"]

    # Remove empty texts to avoid processing errors
    original_count = len(df)
    df = df[df["combined_text"].str.strip() != ""]
    if len(df) < original_count:
        print(f"Removed {original_count - len(df)} empty posts.")

    # --- Pre-filter: Identify texts with required patterns ---
    print("Identifying posts with phone numbers or keywords...")
    detector = HybridSpamDetector()

    # Check which posts have the required patterns
    df["has_phone_number"] = df["combined_text"].apply(contains_phone_number)
    df["has_keyword_with_numbers"] = df["combined_text"].apply(
        contains_keyword_with_numbers
    )
    df["has_specified_keywords"] = df["combined_text"].apply(
        contains_specified_keywords
    )
    df["has_required_patterns"] = (
        df["has_phone_number"]
        | df["has_keyword_with_numbers"]
        | df["has_specified_keywords"]
    )

    pattern_count = df["has_required_patterns"].sum()
    print(
        f"Found {pattern_count} posts with phone numbers or specified keywords ({pattern_count / len(df) * 100:.1f}%)"
    )

    # --- Spam Detection ---
    print("Analyzing posts in batches...")
    texts_to_analyze = df["combined_text"].tolist()

    # Process in smaller batches for better memory management on MPS
    batch_size = 32
    all_results = []

    for i in range(0, len(texts_to_analyze), batch_size):
        batch_texts = texts_to_analyze[i: i + batch_size]
        batch_results = detector.predict_batch(batch_texts)
        all_results.extend(batch_results)

        if (i // batch_size) % 10 == 0:  # Print progress every 10 batches
            print(
                (
                    f"Processed {min(i + batch_size, len(texts_to_analyze))}/{len(texts_to_analyze)} posts..."
                )
            )

    print("Analysis complete.")

    # --- Save Results ---
    df["is_spam"] = [r[0] for r in all_results]
    df["spam_confidence"] = [r[1] for r in all_results]

    # Add pattern breakdown for analysis
    output_df = df[
        [
            "id",
            "site_id",
            "title",
            "is_spam",
            "spam_confidence",
            "has_phone_number",
            "has_keyword_with_numbers",
            "has_specified_keywords",
        ]
    ]
    output_file = "spam_detection_results.csv"
    output_df.to_csv(output_file, index=False)

    # --- Print Summary ---
    spam_count = df["is_spam"].sum()
    total_count = len(df)
    spam_percentage = (spam_count / total_count * 100) if total_count > 0 else 0

    # Breakdown of patterns in spam posts
    spam_df = df[df["is_spam"] == 1]
    phone_in_spam = spam_df["has_phone_number"].sum()
    keyword_num_in_spam = spam_df["has_keyword_with_numbers"].sum()
    keywords_in_spam = spam_df["has_specified_keywords"].sum()

    print("\n" + "=" * 20)
    print("      RESULTS SUMMARY")
    print("=" * 20)
    print(f"Total Posts Analyzed: {total_count}")
    print(
        f"Posts with Phone/Keywords: {pattern_count} ({pattern_count / total_count * 100:.1f}%)"
    )
    print(f"Spam Posts Detected:  {spam_count} ({spam_percentage:.1f}%)")
    print(f"Results saved to:     {output_file}")
    print("\nPatterns in Spam Posts:")
    print(f"  - Contains phone numbers: {phone_in_spam}")
    print(f"  - Contains keywords + numbers: {keyword_num_in_spam}")
    print(f"  - Contains specified keywords: {keywords_in_spam}")
    print("=" * 20)

    # Display some examples of detected spam with pattern info
    print("\nExamples of detected spam posts:")
    spam_examples = output_df[output_df["is_spam"] == 1].head(5)
    if not spam_examples.empty:
        for _, row in spam_examples.iterrows():
            patterns = []
            if row["has_phone_number"]:
                patterns.append("phone")
            if row["has_keyword_with_numbers"]:
                patterns.append("keyword+numbers")
            if row["has_specified_keywords"]:
                patterns.append("keywords")

            print(
                f"ID: {row['id']}, Patterns: {', '.join(patterns)}, Confidence: {row['spam_confidence']:.2f}"
            )
            print(f"Title: {row['title'][:100]}...")
            print()
    else:
        print("No spam posts were detected in this run.")


if __name__ == "__main__":
    main()
