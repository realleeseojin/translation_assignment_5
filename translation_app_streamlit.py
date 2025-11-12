"""
영문 기사 번역 프로그램 (Streamlit UI)
Fine-tuned 모델을 사용한 영어→한국어 번역
"""

# !pip install streamlit transformers torch

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import time
import pandas as pd
from datetime import datetime

# =====================================
# 페이지 설정
# =====================================

st.set_page_config(
    page_title="영문 기사 번역기",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTextArea textarea {
        font-size: 14px;
        font-family: 'Malgun Gothic', sans-serif;
    }
    .translation-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #1f2937;
        border-bottom: 3px solid #4B79A1;
        padding-bottom: 10px;
    }
    .stats-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4B79A1;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================
# 1. 번역 모델 클래스
# =====================================

@st.cache_resource
def load_model(model_path="./final-translation-model"):
    """
    모델을 로드하고 캐싱 (한 번만 로드)
    """
    class TranslationModel:
        def __init__(self, model_path):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            try:
                # 파인튜닝된 모델 로드
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                st.success(f"✅ Fine-tuned 모델 로드 완료")
            except:
                # 원본 모델 로드
                model_name = "Helsinki-NLP/opus-mt-tc-big-en-ko"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                st.info(f"ℹ️ 기본 모델 사용: {model_name}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
        
        def split_into_sentences(self, text):
            """문장 단위로 분할"""
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
        
        def translate_sentence(self, sentence, max_length=256):
            """단일 문장 번역"""
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    temperature=0.9,
                    do_sample=False,
                    early_stopping=True
                )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        def translate_article(self, article_text, progress_callback=None):
            """전체 기사 번역"""
            if not article_text.strip():
                return ""
            
            paragraphs = article_text.split('\n')
            translated_paragraphs = []
            total_sentences = sum(len(self.split_into_sentences(p)) for p in paragraphs if p.strip())
            current_sentence = 0
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    translated_paragraphs.append("")
                    continue
                
                sentences = self.split_into_sentences(paragraph)
                translated_sentences = []
                
                for sentence in sentences:
                    if sentence:
                        translation = self.translate_sentence(sentence)
                        translated_sentences.append(translation)
                        current_sentence += 1
                        
                        # 진행률 콜백
                        if progress_callback:
                            progress_callback(current_sentence / total_sentences)
                
                translated_paragraph = ' '.join(translated_sentences)
                translated_paragraphs.append(translated_paragraph)
            
            return '\n'.join(translated_paragraphs)
    
    return TranslationModel(model_path)

# =====================================
# 2. 메인 UI
# =====================================

# 제목
st.markdown("<h1>📰 영문 기사 번역기</h1>", unsafe_allow_html=True)
st.markdown("**Fine-tuned 모델을 사용한 영어→한국어 기사 번역**")

# 모델 로드
if 'translator' not in st.session_state:
    with st.spinner('모델을 로드하는 중...'):
        st.session_state.translator = load_model()

# 번역 히스토리 초기화
if 'history' not in st.session_state:
    st.session_state.history = []

# 사이드바 - 설정 및 히스토리
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 번역 옵션
    preserve_paragraphs = st.checkbox("단락 구조 유지", value=True)
    show_stats = st.checkbox("통계 정보 표시", value=True)
    
    st.divider()
    
    # 번역 히스토리
    st.header("📋 번역 히스토리")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"{item['time']} - {item['preview']}..."):
                st.text("원문:")
                st.write(item['original'][:200] + "...")
                st.text("번역:")
                st.write(item['translated'][:200] + "...")
    else:
        st.info("아직 번역 기록이 없습니다")
    
    if st.button("기록 삭제"):
        st.session_state.history = []
        st.rerun()

# 메인 컨텐츠 - 2열 레이아웃
col1, col2 = st.columns(2)

with col1:
    st.subheader("영문 기사")
    
    # 텍스트 입력
    english_text = st.text_area(
        "",
        height=400,
        placeholder="번역할 영문 기사를 입력하세요...\n\nEnter English article text here...",
        key="english_input"
    )
    
    # 예제 텍스트
    st.markdown("**📝 예제 기사**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    example_texts = {
        "AI 뉴스": """Breaking News: AI Technology Advances

Artificial intelligence continues to revolutionize various industries. Machine learning models are becoming more sophisticated and accessible to developers worldwide.

Recent developments in natural language processing have led to significant improvements in translation accuracy.""",
        
        "기후 변화": """Climate Change Report

Scientists have released a new report on climate change impacts. The study shows that global temperatures are rising faster than previously predicted.

Immediate action is needed to reduce carbon emissions.""",
        
        "기술 혁신": """Technology Innovation

A new breakthrough in quantum computing has been announced. Researchers claim this could revolutionize data processing speeds.

The technology shows promising results for future applications."""
    }
    
    with col_ex1:
        if st.button("AI 뉴스"):
            st.session_state.english_input = example_texts["AI 뉴스"]
            st.rerun()
    
    with col_ex2:
        if st.button("기후 변화"):
            st.session_state.english_input = example_texts["기후 변화"]
            st.rerun()
    
    with col_ex3:
        if st.button("기술 혁신"):
            st.session_state.english_input = example_texts["기술 혁신"]
            st.rerun()

with col2:
    st.subheader("번역된 한글 기사")
    
    # 번역 결과 표시 영역
    translation_container = st.empty()
    stats_container = st.empty()
    
    # 초기 플레이스홀더
    translation_container.text_area(
        "",
        value="",
        height=400,
        placeholder="번역 결과가 여기에 표시됩니다...",
        key="korean_output",
        disabled=True
    )

# 번역 버튼
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])

with col_btn1:
    if st.button("🔄 번역하기", type="primary", use_container_width=True):
        if english_text:
            # 진행 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"번역 중... {int(progress * 100)}%")
            
            # 번역 시작
            start_time = time.time()
            status_text.text("번역을 시작합니다...")
            
            try:
                # 번역 실행
                translated_text = st.session_state.translator.translate_article(
                    english_text,
                    progress_callback=update_progress
                )
                
                # 번역 시간
                translation_time = time.time() - start_time
                
                # 결과 표시
                with col2:
                    # 번역 결과
                    st.text_area(
                        "",
                        value=translated_text,
                        height=400,
                        key="korean_output_result",
                        disabled=True
                    )
                    
                    # 통계 정보
                    if show_stats:
                        word_count = len(english_text.split())
                        char_count = len(english_text)
                        
                        stats_html = f"""
                        <div class="stats-box">
                            <h4>📊 번역 통계</h4>
                            <ul>
                                <li>원문 단어 수: {word_count}개</li>
                                <li>원문 문자 수: {char_count}자</li>
                                <li>번역 시간: {translation_time:.2f}초</li>
                                <li>평균 속도: {word_count/translation_time:.1f} 단어/초</li>
                            </ul>
                        </div>
                        """
                        st.markdown(stats_html, unsafe_allow_html=True)
                
                # 히스토리에 추가
                st.session_state.history.append({
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'preview': english_text[:30],
                    'original': english_text,
                    'translated': translated_text
                })
                
                # 진행 표시 제거
                progress_bar.empty()
                status_text.success("✅ 번역 완료!")
                
            except Exception as e:
                st.error(f"❌ 번역 중 오류 발생: {str(e)}")
                progress_bar.empty()
                status_text.empty()
        else:
            st.warning("⚠️ 번역할 텍스트를 입력해주세요")

with col_btn2:
    if st.button("🗑️ 지우기", use_container_width=True):
        st.session_state.english_input = ""
        st.session_state.korean_output = ""
        st.rerun()

with col_btn3:
    # 다운로드 버튼 (번역 결과가 있을 때만)
    if 'korean_output_result' in st.session_state and st.session_state.korean_output_result:
        st.download_button(
            label="💾 저장",
            data=st.session_state.korean_output_result,
            file_name=f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# 하단 정보
st.divider()
st.markdown("""
<div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
    <h4>💡 사용 팁</h4>
    <ul>
        <li>긴 기사도 문장 단위로 나누어 정확하게 번역합니다</li>
        <li>단락 구조가 자동으로 유지되어 가독성이 좋습니다</li>
        <li>GPU 사용 시 더 빠른 번역이 가능합니다</li>
    </ul>
    <h4>⚠️ 주의사항</h4>
    <ul>
        <li>전문 용어나 고유명사는 완벽하지 않을 수 있습니다</li>
        <li>매우 긴 문장은 여러 문장으로 나누어 번역하는 것을 권장합니다</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# =====================================
# 3. 실행 방법
# =====================================

# 터미널에서 실행:
# streamlit run translation_app_streamlit.py
