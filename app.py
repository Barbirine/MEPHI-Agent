import logging
import streamlit as st
from config import Settings
from rag.pipeline import build_pipeline, ingest
from rag.uploads import save_uploaded_pdf
from rag.vector_store import VectorStore

st.set_page_config(page_title='MIFI-Agent', page_icon='📚')
st.title('📚 MIFI-Agent')
st.caption('Вопросы по вашим PDF · GigaChat · источники с номерами страниц')
try:
    settings = Settings.load()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if not settings.credentials:
    st.info('Добавьте Authorization key в GIGACHAT_CREDENTIALS в файле .env. '
            'Загрузка и индексация PDF работают без ключа. Ключ нужен для ответов.')
with st.sidebar:
    st.header('Учебные материалы')
    uploads = st.file_uploader('Загрузите методички PDF', type=['pdf'], accept_multiple_files=True)
    st.caption(f'До {settings.max_upload_mb} МБ на файл. PDF должен содержать текстовый слой.')
    if st.button('Сохранить PDF'):
        if not uploads:
            st.warning('Сначала выберите хотя бы один PDF.')
        else:
            for uploaded in uploads:
                try:
                    path = save_uploaded_pdf(uploaded.name, uploaded.getvalue(), settings.pdf_dir,
                                             settings.max_upload_mb * 1024 * 1024)
                    st.success(f'Сохранён: {path.name}')
                except Exception as exc:
                    logging.getLogger(__name__).warning('Ошибка PDF: %s', type(exc).__name__)
                    st.error(f'{uploaded.name}: {exc}')
            st.info('Для поиска по сохранённым файлам нажмите «Индексировать PDF».')
    st.caption('Индексация и поиск выполняются локально. '
               'ответы отправляют вопрос и найденные фрагменты в GigaChat. '
               'Это расходует токены API по вашему тарифу.')
    if st.button('Индексировать PDF', help='Пересоздаёт индекс на этом компьютере.'):
        try:
            with st.spinner('Индексируем PDF локально…'):
                count = ingest(settings)
            st.success(f'Готово: {count} фрагментов. Можно задавать вопросы.')
        except Exception as exc:
            logging.getLogger(__name__).warning('Ошибка индексации: %s', type(exc).__name__)
            st.error(str(exc))
    try:
        meta = VectorStore(settings.index_path).metadata()
        st.caption(f"В индексе: {meta['count']} фрагментов · {meta['model']}")
    except (FileNotFoundError, OSError):
        st.caption('Индекс ещё не создан.')
    except Exception:
        st.warning('Не удалось прочитать индекс. Пересоздайте его.')
with st.form('question'):
    question = st.text_area('Ваш вопрос', max_chars=4000)
    submitted = st.form_submit_button('Найти ответ')
if submitted:
    try:
        if not question.strip():
            raise ValueError('Введите вопрос')
        with st.spinner('Ищем фрагменты и готовим ответ…'):
            answer = build_pipeline(settings).ask(question)
        st.markdown(answer.text)
        for i, hit in enumerate(answer.hits, 1):
            label = 'использован' if i in answer.citation_ids else 'найден'
            with st.expander(f'[{i}] {hit.chunk.document} · PDF-страница {hit.chunk.page} · {label}'):
                st.text(hit.chunk.text)
                st.caption(f'Косинусное сходство: {hit.score:.3f}; это не вероятность правильности.')
        st.caption('Проверьте подтверждение утверждения. Нумерация страниц — от начала PDF.')
    except Exception as exc:
        logging.getLogger(__name__).warning('Ошибка ответа: %s', type(exc).__name__)
        st.error(str(exc))
