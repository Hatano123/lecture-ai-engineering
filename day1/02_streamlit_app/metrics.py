# metrics.py
import streamlit as st
import nltk
from janome.tokenizer import Tokenizer
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import math # logを使う可能性のため（現状未使用）
import unicodedata # 正規化用

# --- NLTK Fallback ---
# NLTKが利用できない場合の簡易代替関数
try:
    nltk.download('punkt', quiet=True)
    from nltk.translate.bleu_score import sentence_bleu as nltk_sentence_bleu
    # NLTKのトークナイザは日本語では使わないので、ここではインポート不要
    print("NLTK loaded successfully for BLEU calculation.") # デバッグ用
except Exception as e:
    print(f"Warning: NLTK initialization failed: {e}. Using fallback BLEU calculation.")
    # NLTKが使えない場合、簡易BLEUスコア（F1スコアに近いもの）を返す
    # 注意: これは実際のBLEUスコアとは異なります
    def nltk_sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25)):
        # Janomeでトークナイズされたリストを受け取ることを想定
        ref_words = set(references[0]) # referencesはリストのリスト [[ref1_token1, ref1_token2], [ref2_token1, ...]]
        cand_words = set(candidate)
        common_words = ref_words.intersection(cand_words)
        precision = len(common_words) / len(cand_words) if cand_words else 0
        recall = len(common_words) / len(ref_words) if ref_words else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # 簡易評価としてF1スコアを返す
        return f1

# --- Janome Tokenizer ---
# Tokenizerインスタンスは一度だけ作成して再利用
try:
    _tokenizer = Tokenizer()
    print("Janome Tokenizer initialized successfully.")
except Exception as e:
    st.error(f"Janome Tokenizer の初期化に失敗しました: {e}")
    _tokenizer = None # Tokenizerが使えない場合はNoneにしておく

# --- Helper Functions ---
def _normalize_text(text: str) -> str:
    """テキストを正規化する（小文字化、記号除去、全角スペースの半角化など）"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKC', text) # 全角英数などを半角に正規化
    text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', ' ', text) # 簡単な記号をスペースに置換
    text = re.sub(r'\s+', ' ', text).strip() # 連続するスペースを一つに
    return text

def _tokenize_text(text: str) -> list[str]:
    """Janomeを使ってテキストを単語（基本形）に分割する"""
    if not _tokenizer or not text:
        return []
    try:
        # 表層形ではなく基本形を使うことで、活用形の差異を吸収する
        tokens = [token.base_form for token in _tokenizer.tokenize(text)]
        return tokens
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return text.split() # エラー時は単純な空白分割にフォールバック

def _calculate_bleu(reference_tokens: list[list[str]], candidate_tokens: list[str]) -> float:
    """BLEUスコアを計算する"""
    if not candidate_tokens or not reference_tokens or not reference_tokens[0]:
        return 0.0
    try:
        # NLTKのsentence_bleuまたはフォールバック関数を使用
        # weightsは4-gramまで考慮
        bleu_score = nltk_sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        return bleu_score
    except ZeroDivisionError:
        # print("Warning: ZeroDivisionError in BLEU calculation (likely short candidate). Returning 0.0")
        return 0.0
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0

def _calculate_cosine_similarity(text1: str, text2: str) -> float:
    """TF-IDFとコサイン類似度を計算する"""
    if not text1 or not text2:
        return 0.0
    try:
        # Janomeでトークナイズした単語リストをスペースで結合した文字列を使う
        tokens1 = _tokenize_text(text1)
        tokens2 = _tokenize_text(text2)
        if not tokens1 or not tokens2:
            return 0.0

        text1_processed = " ".join(tokens1)
        text2_processed = " ".join(tokens2)

        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b') # デフォルトのトークナイザは使わない
        tfidf_matrix = vectorizer.fit_transform([text1_processed, text2_processed])

        # TF-IDFベクトルがゼロベクトルになる場合（例: 語彙に含まれない単語のみ）のエラー回避
        if tfidf_matrix.shape[1] == 0:
             print("Warning: TF-IDF Vectorizer vocabulary is empty. Returning 0.0 similarity.")
             return 0.0

        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity_score
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0

def _calculate_relevance_score(answer_tokens: list[str], correct_tokens: list[str]) -> float:
    """関連性スコア（共通単語の割合）を計算する"""
    if not correct_tokens: # 正解がない場合は計算不能
        return 0.0
    if not answer_tokens: # 回答がない場合は0
        return 0.0
    try:
        answer_words = set(answer_tokens)
        correct_words = set(correct_tokens)
        common_words = answer_words.intersection(correct_words)
        # Jaccard係数に近いが、分母を正解の単語数にする（正解の単語をどれだけカバーできたか）
        relevance = len(common_words) / len(correct_words)
        return relevance
    except Exception as e:
        print(f"Error calculating relevance score: {e}")
        return 0.0

# --- Main Calculation Function ---
def calculate_metrics(answer: str, correct_answer: str):
    """
    回答と正解から評価指標を計算する（改善版）
    Args:
        answer (str): モデルの生成した回答
        correct_answer (str): ユーザーが提供した正解例（または理想的な回答）
    Returns:
        tuple: (bleu_score, similarity_score, word_count, relevance_score)
    """
    # 初期値
    bleu_score = 0.0
    similarity_score = 0.0
    word_count = 0
    relevance_score = 0.0

    # --- 1. 前処理: 正規化とトークナイズ ---
    normalized_answer = _normalize_text(answer)
    answer_tokens = _tokenize_text(normalized_answer)
    word_count = len(answer_tokens) # 単語数はトークン数とする

    # 正解がある場合のみ、正解も処理
    normalized_correct_answer = ""
    correct_tokens = []
    if correct_answer:
        normalized_correct_answer = _normalize_text(correct_answer)
        correct_tokens = _tokenize_text(normalized_correct_answer)

    # --- 2. 各指標の計算 ---
    # 正解がある場合のみBLEU, 類似度, 関連性を計算
    if correct_tokens:
        # BLEU スコア (Janomeトークンを使用)
        reference_tokens_list = [correct_tokens] # sentence_bleuは参照リストのリストを期待
        bleu_score = _calculate_bleu(reference_tokens_list, answer_tokens)

        # コサイン類似度 (Janomeトークンを元にしたTF-IDF)
        # 元の正規化済みテキストを使う（TF-IDF内部の処理のため）
        similarity_score = _calculate_cosine_similarity(normalized_answer, normalized_correct_answer)

        # 関連性スコア (Janomeトークンを使用)
        relevance_score = _calculate_relevance_score(answer_tokens, correct_tokens)

    # --- 3. 結果を返す ---
    return bleu_score, similarity_score, word_count, relevance_score


# --- NLTK データチェック関数 ---
def initialize_nltk():
    """NLTKのデータダウンロードを試みる関数（変更なし）"""
    try:
        nltk.download('punkt', quiet=True)
        print("NLTK Punkt data checked/downloaded.")
    except Exception as e:
        # ここはStreamlitアプリ起動時の情報としてst.errorでも良いかもしれない
        print(f"Error: Failed to download NLTK data: {e}")
        st.error(f"NLTKデータのダウンロードに失敗しました: {e}")

# --- 指標説明関数 ---
def get_metrics_descriptions():
    """評価指標の説明を返す（内容は変更なし、一部追記）"""
    return {
        "正確性スコア (is_correct)": "回答の正確さを3段階で評価: 1.0 (正確), 0.5 (部分的に正確), 0.0 (不正確)",
        "応答時間 (response_time)": "質問を投げてから回答を得るまでの時間（秒）。モデルの効率性を表す",
        "BLEU スコア (bleu_score)": "機械翻訳評価指標で、正解と回答のn-gramの一致度を測定 (0〜1の値、高いほど類似)。Janomeでトークナイズした結果で計算。",
        "類似度スコア (similarity_score)": "TF-IDFベクトル（Janomeトークンベース）のコサイン類似度による、正解と回答の意味的な類似性 (0〜1の値)。",
        "単語数 (word_count)": "回答に含まれる単語（Janomeによる形態素解析後の基本形）の数。情報量や詳細さの指標。",
        "関連性スコア (relevance_score)": "正解と回答の共通単語（Janomeトークン）の割合（正解単語数に対する比率）。トピックの関連性を表す (0〜1の値)。",
        "効率性スコア (efficiency_score)": "正確性を応答時間で割った値。高速で正確な回答ほど高スコア (UI側で計算)。"
    }

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # テスト用の例文
    test_answer = "私は昨日美味しいラーメンを食べました。"
    test_correct = "昨日、私は美味しいラーメンを食べた。"
    test_answer_en = "the cat sat on the mat"
    test_correct_en = "the cat was on the mat"

    print("--- Japanese Test ---")
    if _tokenizer:
        bleu, sim, wc, rel = calculate_metrics(test_answer, test_correct)
        print(f"回答: {test_answer}")
        print(f"正解: {test_correct}")
        print(f"BLEU: {bleu:.4f}")
        print(f"Similarity: {sim:.4f}")
        print(f"Word Count: {wc}")
        print(f"Relevance: {rel:.4f}")
    else:
        print("Janome Tokenizer not available. Skipping Japanese test.")

    # print("\n--- English Test (using fallback BLEU if NLTK fails) ---")
    # Note: Janome is not ideal for English, TfidfVectorizer/NLTK tokenizer would be better
    # bleu_en, sim_en, wc_en, rel_en = calculate_metrics(test_answer_en, test_correct_en)
    # print(f"Answer: {test_answer_en}")
    # print(f"Correct: {test_correct_en}")
    # print(f"BLEU: {bleu_en:.4f}")
    # print(f"Similarity: {sim_en:.4f}")
    # print(f"Word Count: {wc_en}")
    # print(f"Relevance: {rel_en:.4f}")

    print("\n--- Metrics Descriptions ---")
    descriptions = get_metrics_descriptions()
    for k, v in descriptions.items():
        print(f"- {k}: {v}")