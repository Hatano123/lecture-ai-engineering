# ui.py
import streamlit as st
import pandas as pd
import time
import uuid
from database import save_to_db, get_chat_history, get_db_count, clear_db
from llm import generate_response
from data import create_sample_evaluation_data
from metrics import get_metrics_descriptions

def initialize_chat_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_feedback_message_id" not in st.session_state:
        st.session_state.current_feedback_message_id = None

def display_message_with_feedback(msg):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if 'response_time' in msg:
                st.caption(f"応答時間: {msg.get('response_time', 0.0):.2f}秒")

            if not msg.get("feedback_submitted", False):
                feedback_cols = st.columns([1, 1, 1, 5])

                with feedback_cols[0]:
                    if st.button("👍", key=f"thumb_up_{msg['id']}", help="この回答は良かった"):
                        submit_feedback(
                            message_id=msg['id'],
                            accuracy_label="正確",
                            feedback_comment="👍 良かった (簡易フィードバック)",
                            correct_answer_text=""
                        )
                        st.rerun()
                with feedback_cols[1]:
                    if st.button("👎", key=f"thumb_down_{msg['id']}", help="この回答は改善が必要"):
                        submit_feedback(
                            message_id=msg['id'],
                            accuracy_label="不正確",
                            feedback_comment="👎 改善が必要 (簡易フィードバック)",
                            correct_answer_text=""
                        )
                        st.rerun()
                with feedback_cols[2]:
                    if st.button("詳細", key=f"detail_feedback_btn_{msg['id']}", help="詳細なフィードバックを行う"):
                        st.session_state.current_feedback_message_id = msg['id']
                        st.rerun()
            else:
                st.success("✔️ フィードバック送信済み")

def display_and_process_detailed_feedback(message_id):
    msg_to_feedback = next((m for m in st.session_state.messages if m['id'] == message_id), None)
    if not msg_to_feedback:
        st.error("フィードバック対象のメッセージが見つかりません。")
        st.session_state.current_feedback_message_id = None
        return

    with st.form(key=f"feedback_form_{message_id}"):
        st.markdown(f"##### 「{msg_to_feedback['content'][:50]}...」への詳細フィードバック")
        feedback_options = ["正確", "部分的に正確", "不正確"]
        feedback_accuracy = st.radio(
            "回答の評価:",
            feedback_options,
            index=0,
            key=f"feedback_radio_{message_id}",
            horizontal=True
        )
        correct_answer_text = st.text_area(
            "より正確な回答（任意）:",
            key=f"correct_answer_input_{message_id}",
            height=100
        )
        feedback_comment_text = st.text_area(
            "コメント（任意）:",
            key=f"feedback_comment_input_{message_id}",
            height=100
        )
        submitted = st.form_submit_button("フィードバックを送信")

        if submitted:
            submit_feedback(
                message_id=message_id,
                accuracy_label=feedback_accuracy,
                feedback_comment=feedback_comment_text,
                correct_answer_text=correct_answer_text
            )
            st.session_state.current_feedback_message_id = None
            st.rerun()

def submit_feedback(message_id, accuracy_label, feedback_comment, correct_answer_text):
    msg_to_feedback = next((m for m in st.session_state.messages if m['id'] == message_id), None)
    if not msg_to_feedback:
        st.error("フィードバック対象のメッセージが見つかりません。")
        return

    is_correct_value = 1.0 if accuracy_label == "正確" else (0.5 if accuracy_label == "部分的に正確" else 0.0)
    combined_feedback_text = f"{accuracy_label}"
    if feedback_comment:
        combined_feedback_text += f": {feedback_comment}"

    save_to_db(
        question=msg_to_feedback.get("original_question_for_assistant", "N/A"),
        answer=msg_to_feedback['content'],
        feedback=combined_feedback_text,
        correct_answer=correct_answer_text,
        is_correct=is_correct_value,
        response_time=msg_to_feedback.get('response_time', 0.0)
    )
    msg_to_feedback["feedback_submitted"] = True
    st.success("フィードバックが保存されました！")

def display_chat_page(pipe):
    initialize_chat_session()
    st.subheader("AIチャット")

    for msg in st.session_state.messages:
        display_message_with_feedback(msg)

    if st.session_state.current_feedback_message_id:
        with st.expander("詳細フィードバック入力", expanded=True):
             display_and_process_detailed_feedback(st.session_state.current_feedback_message_id)
             if st.button("詳細フィードバックを閉じる", key=f"close_detail_fb_form_{st.session_state.current_feedback_message_id}"):
                 st.session_state.current_feedback_message_id = None
                 st.rerun()

    user_prompt = st.chat_input("メッセージを入力してください...")

    if user_prompt:
        user_msg_id = str(uuid.uuid4())
        st.session_state.messages.append({"id": user_msg_id, "role": "user", "content": user_prompt})

        with st.spinner("AIが考えています..."):
            ai_response_content, response_time_val = generate_response(pipe, user_prompt)

        ai_msg_id = str(uuid.uuid4())
        st.session_state.messages.append({
            "id": ai_msg_id,
            "role": "assistant",
            "content": ai_response_content,
            "response_time": response_time_val,
            "feedback_submitted": False,
            "original_question_for_assistant": user_prompt
        })
        st.rerun()

def display_history_page():
    st.subheader("チャット履歴と評価指標")
    history_df = get_chat_history()

    if history_df.empty:
        st.info("まだチャット履歴がありません。")
        return

    tab1, tab2 = st.tabs(["履歴閲覧", "評価指標分析"])
    with tab1:
        display_history_list(history_df)
    with tab2:
        display_metrics_analysis(history_df)

def display_history_list(history_df):
    st.write("#### 履歴リスト")
    filter_options = {"すべて表示": None, "正確なもののみ": 1.0, "部分的に正確なもののみ": 0.5, "不正確なもののみ": 0.0}
    display_option = st.radio("表示フィルタ", options=list(filter_options.keys()), horizontal=True, label_visibility="collapsed", key="history_filter_radio")

    filter_value = filter_options[display_option]
    if filter_value is not None:
        filtered_df = history_df[history_df["is_correct"].notna() & (history_df["is_correct"] == filter_value)]
    else:
        filtered_df = history_df

    if filtered_df.empty:
        st.info("選択した条件に一致する履歴はありません。")
        return

    items_per_page = 5
    total_items = len(filtered_df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    total_pages = max(1, total_pages)
    current_page = st.number_input('ページ', min_value=1, max_value=total_pages, value=1, step=1, key="history_page_num_input")

    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paginated_df = filtered_df.iloc[start_idx:end_idx]

    for i, row in paginated_df.iterrows():
        q_text = str(row['question']) if pd.notna(row['question']) else 'N/A'
        with st.expander(f"{row['timestamp']} - Q: {q_text[:50]}..."):
            st.markdown(f"**Q:** {q_text}")
            st.markdown(f"**A:** {str(row['answer']) if pd.notna(row['answer']) else 'N/A'}")
            st.markdown(f"**Feedback:** {str(row['feedback']) if pd.notna(row['feedback']) else 'N/A'}")
            if pd.notna(row['correct_answer']) and row['correct_answer']:
                st.markdown(f"**Correct A:** {row['correct_answer']}")
            st.markdown("---")
            m_cols = st.columns(3)
            m_cols[0].metric("正確性", f"{row['is_correct']:.1f}" if pd.notna(row['is_correct']) else "-")
            m_cols[1].metric("応答時間(s)", f"{row['response_time']:.2f}" if pd.notna(row['response_time']) else "-")
            m_cols[2].metric("単語数", f"{int(row['word_count'])}" if pd.notna(row['word_count']) else "-")
            m_cols_2 = st.columns(3)
            m_cols_2[0].metric("BLEU", f"{row['bleu_score']:.4f}" if pd.notna(row['bleu_score']) else "-")
            m_cols_2[1].metric("類似度", f"{row['similarity_score']:.4f}" if pd.notna(row['similarity_score']) else "-")
            m_cols_2[2].metric("関連性", f"{row['relevance_score']:.4f}" if pd.notna(row['relevance_score']) else "-")

    st.caption(f"{total_items} 件中 {start_idx+1 if total_items > 0 else 0} - {min(end_idx, total_items)} 件を表示")

def display_metrics_analysis(history_df):
    st.write("#### 評価指標の分析")
    analysis_df = history_df.dropna(subset=['is_correct']).copy()
    if analysis_df.empty:
        st.warning("分析可能な評価データがありません。")
        return

    accuracy_labels = {1.0: '正確', 0.5: '部分的に正確', 0.0: '不正確'}
    analysis_df.loc[:, '正確性ラベル'] = analysis_df['is_correct'].map(accuracy_labels)

    st.write("##### 正確性の分布")
    if '正確性ラベル' in analysis_df.columns:
        accuracy_counts = analysis_df['正確性ラベル'].value_counts()
        if not accuracy_counts.empty: st.bar_chart(accuracy_counts)
        else: st.info("正確性データがありません。")
    else: st.info("正確性ラベルカラムが分析データにありません。")

    st.write("##### 応答時間とその他の指標の関係")
    metric_options = ["bleu_score", "similarity_score", "relevance_score", "word_count"]
    valid_metric_options = [m for m in metric_options if m in analysis_df.columns and analysis_df[m].notna().any()]

    if valid_metric_options:
        selected_metric = st.selectbox("比較する評価指標を選択", valid_metric_options, key="analysis_metric_selectbox")
        plot_df = analysis_df[['response_time', selected_metric, '正確性ラベル']].dropna()
        if not plot_df.empty:
            st.scatter_chart(plot_df, x='response_time', y=selected_metric, color='正確性ラベル')
        else: st.info(f"選択された指標 ({selected_metric}) と応答時間の有効なデータがありません。")
    else: st.info("応答時間と比較できる指標データがありません。")

    st.write("##### 評価指標の統計")
    stats_cols = ['response_time', 'bleu_score', 'similarity_score', 'word_count', 'relevance_score']
    valid_stats_cols = [
        c for c in stats_cols
        if c in analysis_df.columns and pd.api.types.is_numeric_dtype(analysis_df[c]) and analysis_df[c].notna().any()
    ]
    if valid_stats_cols:
        st.dataframe(analysis_df[valid_stats_cols].describe())
    else: st.info("統計情報を計算できる数値型の評価指標データがありません。")

    st.write("##### 正確性レベル別の平均スコア")
    if valid_stats_cols and '正確性ラベル' in analysis_df.columns:
        try:
            grouped_stats = analysis_df.groupby('正確性ラベル')[valid_stats_cols].mean()
            st.dataframe(grouped_stats)
        except Exception as e: st.warning(f"正確性別スコアの集計中にエラーが発生しました: {e}")
    else: st.info("正確性レベル別の平均スコアを計算できるデータがありません。")

    st.write("##### 効率性スコア (正確性 / (応答時間 + 0.1))")
    if ('response_time' in analysis_df.columns and analysis_df['response_time'].notna().any() and
        'is_correct' in analysis_df.columns and analysis_df['is_correct'].notna().any()):
        analysis_df.loc[:, 'efficiency_score'] = analysis_df['is_correct'] / (analysis_df['response_time'].fillna(0) + 0.1)
        top_efficiency = analysis_df.sort_values('efficiency_score', ascending=False).head(10)
        if not top_efficiency.empty:
            st.bar_chart(top_efficiency['efficiency_score'].reset_index(drop=True))
        else: st.info("効率性スコアデータがありません。")
    else: st.info("効率性スコアを計算するための応答時間または正確性データが不足しています。")

def display_data_page():
    st.subheader("サンプル評価データの管理")
    count = get_db_count()
    st.write(f"現在のデータベースには {count} 件のレコードがあります。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("サンプルデータを追加", key="add_sample_data_page_btn"):
            create_sample_evaluation_data()
            st.rerun()
    with col2:
        if st.button("データベースをクリア", key="clear_db_data_page_btn"):
            if clear_db():
                st.rerun()

    st.subheader("評価指標の説明")
    metrics_info = get_metrics_descriptions()
    for metric_name, description in metrics_info.items():
        with st.expander(f"{metric_name}"):
            st.write(description)