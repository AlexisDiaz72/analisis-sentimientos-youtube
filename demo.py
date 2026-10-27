import argparse
import os
import pickle
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from googleapiclient.errors import HttpError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob

from Google import Create_Service

API_NAME = 'youtube'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
MAX_COMMENTS_ALLOWED = 100
DEFAULT_VIDEO_IDS = (
    'BNIws4K9YmM, dmRukPHV-kI, 2eQGcYotnp0, 9agX3gY1jU, RgYEPZ4qdGM, MZEZuMjlBOI, '
    'PIuQTWzLlYg, yqqjKg1kEUk, XXQQutEaaV8, WIb4B79B9zQ, i9IkctQFNX4, UxUybYT_WZo, '
    'ItpY33E8JHk, jwMlZNVn-3k, d8OI4kX8TMM, qWS6lZqdfu4, rn-GdtMoaRQ, Pp6hEHQjZoc, '
    '_mOG0SL4tqo, ln1JTOJ4dyU, YXJ6ifo1xXE, dXO6f7AhIxY, vJmg4EiCJj4, zUmQDGt4YFg, '
    'CjxApgXkDZ8, Mf_mRF6M_Jk, Jd5V6CBHP78, 3p-5Qe0rDzk, hZlpFAo2-w4, -hPa1GcxLxU, '
    'WqUML62md8w, yMlzRdTRq_A'
)
SPANISH_STOPWORDS = {
    'a', 'acaba', 'ahi', 'al', 'algo', 'algun', 'alguna', 'algunas', 'alguno', 'algunos', 'alla',
    'alli', 'ambos', 'ante', 'antes', 'aquel', 'aquella', 'aquellas', 'aquello', 'aquellos', 'aqui',
    'arriba', 'asi', 'aun', 'aunque', 'bajo', 'bien', 'cada', 'casi', 'como', 'con', 'contra',
    'cual', 'cuales', 'cualquier', 'cuando', 'de', 'debe', 'deben', 'del', 'desde', 'donde', 'dos',
    'el', 'ella', 'ellas', 'ello', 'ellos', 'en', 'entre', 'era', 'erais', 'eran', 'eras', 'eres',
    'es', 'esa', 'esas', 'ese', 'eso', 'esos', 'esta', 'estaba', 'estabais', 'estaban', 'estabas',
    'estad', 'estada', 'estadas', 'estado', 'estados', 'estais', 'estamos', 'estan', 'estando',
    'estar', 'estaremos', 'estara', 'estaran', 'estaras', 'estare', 'estareis', 'estaria',
    'estariais', 'estariamos', 'estarian', 'estarias', 'estas', 'este', 'estemos', 'esto', 'estos',
    'estoy', 'fue', 'fueron', 'fui', 'fuimos', 'ha', 'hace', 'hacen', 'hacer', 'hacia', 'han', 'hasta',
    'hay', 'la', 'las', 'le', 'les', 'lo', 'los', 'mas', 'me', 'mi', 'mientras', 'mis', 'mucho',
    'muy', 'nada', 'ni', 'no', 'nos', 'nosotros', 'nuestra', 'nuestras', 'nuestro', 'nuestros',
    'o', 'os', 'otra', 'otras', 'otro', 'otros', 'para', 'pero', 'poco', 'por', 'porque', 'que',
    'quien', 'quienes', 'se', 'sea', 'segun', 'ser', 'si', 'siempre', 'sin', 'sobre', 'sois', 'solo',
    'somos', 'son', 'soy', 'su', 'sus', 'tal', 'tambien', 'te', 'teneis', 'tenemos', 'tener', 'tengo',
    'ti', 'tiene', 'tienen', 'todo', 'todos', 'tu', 'tus', 'un', 'una', 'uno', 'unos', 'usa', 'uso',
    'usted', 'ustedes', 'va', 'vamos', 'van', 'varias', 'varios', 'vosotras', 'vosotros', 'y', 'ya',
    'yo'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Mineria de opiniones para videos de YouTube.')
    parser.add_argument('--video-id', default='', help='ID unico de YouTube a analizar (opcional).')
    parser.add_argument(
        '--video-ids',
        nargs='+',
        default=[DEFAULT_VIDEO_IDS],
        help='Lista de IDs de YouTube a analizar. Acepta comas o espacios.'
    )
    parser.add_argument(
        '--max-comments',
        type=int,
        default=100,
        help='Numero maximo de comentarios a descargar. Tope absoluto: 100.'
    )
    parser.add_argument('--db-path', default='youtube_mineria.db', help='Ruta del archivo SQLite.')
    parser.add_argument('--excel-path', default='mineria_buap_final.xlsx', help='Ruta del Excel de salida.')
    parser.add_argument('--csv-path', default='mineria_buap_final.csv', help='Ruta del CSV de salida.')
    parser.add_argument('--plot-path', default='grafica_sentimientos.png', help='Ruta de la grafica de barras.')
    parser.add_argument('--pdf-path', default='reporte_mineria.pdf', help='Ruta del reporte PDF ejecutivo.')
    parser.add_argument(
        '--sentiment-model',
        choices=['logreg', 'textblob'],
        default='logreg',
        help='Modelo de sentimiento a usar.'
    )
    parser.add_argument('--model-path', default='sentiment_logreg_tfidf.pkl', help='Ruta del modelo persistido.')
    parser.add_argument('--train-data-path', default='', help='CSV de entrenamiento con columnas text/sentiment (opcional).')
    parser.add_argument('--retrain-model', action='store_true', help='Reentrena el modelo ignorando cache.')
    return parser.parse_args()


def normalize_video_ids(args):
    raw_values = args.video_ids if args.video_ids else [args.video_id]
    video_ids = []
    for value in raw_values:
        normalized_value = str(value).replace('.', ' ')
        for part in re.split(r'[\s,]+', normalized_value):
            cleaned = part.strip()
            if cleaned:
                video_ids.append(cleaned)

    return list(dict.fromkeys(video_ids))

def resolve_client_secret_file():
    env_path = os.getenv('YOUTUBE_CLIENT_SECRET')
    candidates = [
        env_path,
        'client_secret.json',
        'client-secret.json',
        'Client_Secret.json',
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        'No se encontro archivo de credenciales. Configura YOUTUBE_CLIENT_SECRET o agrega client-secret.json.'
    )

def execute_with_retry(request, max_retries=3, base_wait=1.5):
    for attempt in range(max_retries):
        try:
            return request.execute()
        except HttpError as error:
            status = getattr(error.resp, 'status', None)
            should_retry = status in {403, 429, 500, 502, 503, 504}
            if not should_retry or attempt == max_retries - 1:
                raise
            sleep_seconds = base_wait * (2 ** attempt)
            time.sleep(sleep_seconds)

def get_video_details(service, video_id):
    response = execute_with_retry(
        service.videos().list(
            part='snippet,statistics,contentDetails,status,topicDetails,liveStreamingDetails,localizations',
            id=video_id,
        )
    )
    items = response.get('items', [])
    if not items:
        raise ValueError(f'No se encontro informacion para el video {video_id}.')
    return items[0]


def get_channel_details(service, channel_id):
    response = execute_with_retry(
        service.channels().list(
            part='snippet,statistics,brandingSettings',
            id=channel_id,
        )
    )
    items = response.get('items', [])
    return items[0] if items else {}


def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positivo'
    if polarity < 0:
        return 'Negativo'
    return 'Neutro'


def textblob_sentiment(text):
    polarity = float(TextBlob(str(text or '')).sentiment.polarity)
    return polarity, classify_sentiment(polarity)


def clean_comment_text(text):
    raw_text = str(text or '')
    cleaned = raw_text.lower()
    cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)
    cleaned = re.sub(r'[@#]\w+', ' ', cleaned)
    cleaned = re.sub(r'[^a-z0-9\sáéíóúñü]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_keywords(series, top_n=20):
    tokens = []
    for value in series.fillna(''):
        for token in str(value).split():
            if len(token) < 3:
                continue
            if token in SPANISH_STOPWORDS:
                continue
            tokens.append(token)

    counts = Counter(tokens)
    return pd.DataFrame(counts.most_common(top_n), columns=['keyword', 'count'])


def enrich_comment_features(comments_df):
    if comments_df.empty:
        return comments_df

    df = comments_df.copy()
    df['comment_text_raw'] = df['comment_text'].fillna('').astype(str)
    df['comment_text_clean'] = df['comment_text_raw'].apply(clean_comment_text)
    df['comment_length'] = df['comment_text_raw'].str.len()
    df['word_count'] = df['comment_text_clean'].str.split().str.len().fillna(0).astype(int)
    df['has_question'] = df['comment_text_raw'].str.contains(r'[?¿]', regex=True)
    df['has_link'] = df['comment_text_raw'].str.contains(r'https?://|www\.', regex=True)

    dt = pd.to_datetime(df['comment_published_at'], errors='coerce', utc=True).dt.tz_localize(None)
    df['comment_datetime'] = dt
    df['comment_date'] = dt.dt.date.astype(str)
    df['comment_hour'] = dt.dt.hour.fillna(-1).astype(int)
    df['comment_weekday'] = dt.dt.day_name().fillna('Desconocido')

    df['is_duplicate_text'] = df.duplicated(subset=['video_id', 'comment_text_clean'], keep=False)
    return df


def _load_training_data(train_data_path):
    if not train_data_path or not os.path.exists(train_data_path):
        return pd.DataFrame()

    df = pd.read_csv(train_data_path)
    text_col = None
    sentiment_col = None

    for candidate in ['text', 'comment_text', 'comment_text_raw', 'comment_text_clean']:
        if candidate in df.columns:
            text_col = candidate
            break

    for candidate in ['sentiment', 'label', 'target']:
        if candidate in df.columns:
            sentiment_col = candidate
            break

    if text_col is None or sentiment_col is None:
        raise ValueError('El CSV de entrenamiento debe contener columnas de texto y sentiment/label/target.')

    train_df = df[[text_col, sentiment_col]].copy()
    train_df.columns = ['text', 'sentiment']
    train_df['text'] = train_df['text'].fillna('').astype(str)
    train_df['sentiment'] = train_df['sentiment'].fillna('').astype(str).str.strip().str.title()
    train_df = train_df[train_df['sentiment'].isin(['Positivo', 'Neutro', 'Negativo'])]
    train_df['text'] = train_df['text'].apply(clean_comment_text)
    train_df = train_df[train_df['text'].str.len() > 0]
    return train_df


def _build_weak_training_data(comments_df):
    train_df = comments_df[['comment_text_clean', 'sentiment']].copy()
    train_df.columns = ['text', 'sentiment']
    train_df['text'] = train_df['text'].fillna('').astype(str)
    train_df['sentiment'] = train_df['sentiment'].fillna('').astype(str)
    train_df = train_df[(train_df['text'].str.len() > 0) & (train_df['sentiment'].isin(['Positivo', 'Neutro', 'Negativo']))]
    return train_df


def train_or_load_logreg_model(args, comments_df):
    if not args.retrain_model and os.path.exists(args.model_path):
        with open(args.model_path, 'rb') as f:
            return pickle.load(f)

    training_df = _load_training_data(args.train_data_path)
    if training_df.empty:
        training_df = _build_weak_training_data(comments_df)

    if training_df.empty or training_df['sentiment'].nunique() < 2:
        raise ValueError('No hay suficientes datos etiquetados para entrenar Regresion Logistica.')

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(training_df['text'])
    y_train = training_df['sentiment']

    model = LogisticRegression(
        max_iter=1200,
        class_weight='balanced',
        solver='lbfgs',
        multi_class='auto',
    )
    model.fit(x_train, y_train)

    model_bundle = {
        'vectorizer': vectorizer,
        'model': model,
        'classes': list(model.classes_),
        'trained_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    }

    with open(args.model_path, 'wb') as f:
        pickle.dump(model_bundle, f)

    return model_bundle


def apply_sentiment_model(comments_df, args):
    if comments_df.empty:
        return comments_df

    df = comments_df.copy()

    if args.sentiment_model == 'textblob':
        scores_labels = df['comment_text_raw'].apply(textblob_sentiment)
        df['polarity'] = scores_labels.apply(lambda x: x[0])
        df['sentiment'] = scores_labels.apply(lambda x: x[1])
        return df

    try:
        bundle = train_or_load_logreg_model(args, df)
        vectorizer = bundle['vectorizer']
        model = bundle['model']

        x_pred = vectorizer.transform(df['comment_text_clean'].fillna(''))
        predicted = model.predict(x_pred)
        probabilities = model.predict_proba(x_pred)
        class_list = list(model.classes_)

        positive_idx = class_list.index('Positivo') if 'Positivo' in class_list else None
        negative_idx = class_list.index('Negativo') if 'Negativo' in class_list else None

        if positive_idx is not None and negative_idx is not None:
            polarity_scores = probabilities[:, positive_idx] - probabilities[:, negative_idx]
        else:
            # Fallback simple si no estan ambas clases en entrenamiento.
            polarity_scores = pd.Series(predicted).map({'Positivo': 1.0, 'Neutro': 0.0, 'Negativo': -1.0}).fillna(0.0).values

        df['sentiment'] = predicted
        df['polarity'] = polarity_scores
    except Exception as error:  # pylint: disable=broad-except
        print(f'Advertencia: fallo Regresion Logistica ({error}). Se usa TextBlob como respaldo.')
        scores_labels = df['comment_text_raw'].apply(textblob_sentiment)
        df['polarity'] = scores_labels.apply(lambda x: x[0])
        df['sentiment'] = scores_labels.apply(lambda x: x[1])

    return df


def fetch_comments(service, video_id, max_comments):
    comments = []
    page_token = None
    remaining = max_comments

    while remaining > 0:
        page_size = min(100, remaining)
        response = execute_with_retry(
            service.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=page_size,
                pageToken=page_token,
                textFormat='plainText',
                order='time',
            )
        )

        for item in response.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            text = snippet.get('textDisplay', '')
            polarity, sentiment = textblob_sentiment(text)
            comments.append(
                {
                    'comment_id': item['snippet']['topLevelComment']['id'],
                    'video_id': video_id,
                    'author_name': snippet.get('authorDisplayName'),
                    'author_channel_id': (snippet.get('authorChannelId') or {}).get('value'),
                    'comment_text': text,
                    'comment_like_count': snippet.get('likeCount', 0),
                    'comment_published_at': snippet.get('publishedAt'),
                    'comment_updated_at': snippet.get('updatedAt'),
                    'reply_count': item['snippet'].get('totalReplyCount', 0),
                    'polarity': polarity,
                    'sentiment': sentiment,
                }
            )

            if len(comments) >= max_comments:
                break

        if len(comments) >= max_comments:
            break

        page_token = response.get('nextPageToken')
        if not page_token:
            break

        remaining = max_comments - len(comments)

    return comments


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_summary(video_item, channel_item, comments_df):
    snippet = video_item.get('snippet', {})
    statistics = video_item.get('statistics', {})
    content_details = video_item.get('contentDetails', {})
    status = video_item.get('status', {})

    channel_snippet = channel_item.get('snippet', {})
    channel_stats = channel_item.get('statistics', {})

    view_count = safe_int(statistics.get('viewCount'))
    like_count = safe_int(statistics.get('likeCount'))
    comment_count_youtube = safe_int(statistics.get('commentCount'))
    comments_collected = len(comments_df)

    engagement_rate = (like_count + comment_count_youtube) / view_count if view_count else 0
    like_view_ratio = like_count / view_count if view_count else 0
    comment_view_ratio = comment_count_youtube / view_count if view_count else 0

    weighted_polarity = 0.0
    if not comments_df.empty:
        weights = comments_df['comment_like_count'].fillna(0) + 1
        weighted_polarity = (comments_df['polarity'] * weights).sum() / weights.sum()

    sentiment_counts = comments_df['sentiment'].value_counts().to_dict() if not comments_df.empty else {}
    positive_count = int(sentiment_counts.get('Positivo', 0))
    neutral_count = int(sentiment_counts.get('Neutro', 0))
    negative_count = int(sentiment_counts.get('Negativo', 0))

    positive_pct = (positive_count / comments_collected * 100.0) if comments_collected else 0.0
    neutral_pct = (neutral_count / comments_collected * 100.0) if comments_collected else 0.0
    negative_pct = (negative_count / comments_collected * 100.0) if comments_collected else 0.0
    satisfaction_index = ((positive_count - negative_count) / comments_collected) if comments_collected else 0.0

    top_keywords_df = extract_keywords(comments_df['comment_text_clean'] if 'comment_text_clean' in comments_df else pd.Series(dtype=str), top_n=8)
    top_keywords = ', '.join(top_keywords_df['keyword'].tolist())

    return {
        'run_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'video_id': video_item.get('id'),
        'video_title': snippet.get('title'),
        'video_description': snippet.get('description'),
        'video_published_at': snippet.get('publishedAt'),
        'video_channel_id': snippet.get('channelId'),
        'video_channel_title': snippet.get('channelTitle'),
        'video_category_id': snippet.get('categoryId'),
        'video_default_language': snippet.get('defaultLanguage'),
        'video_default_audio_language': snippet.get('defaultAudioLanguage'),
        'video_duration_iso8601': content_details.get('duration'),
        'video_definition': content_details.get('definition'),
        'video_caption': content_details.get('caption'),
        'video_licensed_content': content_details.get('licensedContent'),
        'video_embeddable': status.get('embeddable'),
        'video_made_for_kids': status.get('madeForKids'),
        'video_privacy_status': status.get('privacyStatus'),
        'video_view_count': view_count,
        'video_like_count': like_count,
        'video_comment_count': comment_count_youtube,
        'channel_title': channel_snippet.get('title'),
        'channel_published_at': channel_snippet.get('publishedAt'),
        'channel_country': channel_snippet.get('country'),
        'channel_subscriber_count': safe_int(channel_stats.get('subscriberCount')),
        'channel_video_count': safe_int(channel_stats.get('videoCount')),
        'channel_view_count': safe_int(channel_stats.get('viewCount')),
        'comments_collected': comments_collected,
        'engagement_rate': engagement_rate,
        'like_view_ratio': like_view_ratio,
        'comment_view_ratio': comment_view_ratio,
        'avg_comment_polarity': float(comments_df['polarity'].mean()) if not comments_df.empty else 0.0,
        'weighted_comment_polarity': float(weighted_polarity),
        'positive_comments': positive_count,
        'neutral_comments': neutral_count,
        'negative_comments': negative_count,
        'positive_comments_pct': positive_pct,
        'neutral_comments_pct': neutral_pct,
        'negative_comments_pct': negative_pct,
        'satisfaction_index': satisfaction_index,
        'top_keywords': top_keywords,
    }

def _strip_tz(df):
    """Convierte columnas datetime con zona horaria a naive para compatibilidad con Excel."""
    df = df.copy()
    for col in df.select_dtypes(include=['datetimetz']).columns:
        df[col] = df[col].dt.tz_localize(None)
    return df


def save_outputs(comments_df, summary_df, excel_path, csv_path, plot_path):

    comments_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    comments_excel = _strip_tz(comments_df)
    summary_excel = _strip_tz(summary_df)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        comments_excel.to_excel(writer, sheet_name='comentarios', index=False)
        summary_excel.to_excel(writer, sheet_name='resumen', index=False)

        if not summary_excel.empty:
            comparison_columns = [
                'video_id',
                'video_title',
                'comments_collected',
                'engagement_rate',
                'avg_comment_polarity',
                'positive_comments_pct',
                'neutral_comments_pct',
                'negative_comments_pct',
                'satisfaction_index',
                'top_keywords',
            ]
            comparison_df = summary_excel[comparison_columns].sort_values(
                by=['satisfaction_index', 'engagement_rate'], ascending=[False, False]
            )
            comparison_df.to_excel(writer, sheet_name='comparativo_videos', index=False)

        if not comments_excel.empty and 'comment_text_clean' in comments_excel:
            keywords_df = extract_keywords(comments_excel['comment_text_clean'], top_n=30)
            keywords_df.to_excel(writer, sheet_name='palabras_clave', index=False)

            keywords_by_sentiment = []
            for sentiment in ['Positivo', 'Neutro', 'Negativo']:
                subset = comments_excel[comments_excel['sentiment'] == sentiment]
                partial = extract_keywords(subset['comment_text_clean'], top_n=15)
                if not partial.empty:
                    partial['sentiment'] = sentiment
                    keywords_by_sentiment.append(partial)

            if keywords_by_sentiment:
                keywords_sentiment_df = pd.concat(keywords_by_sentiment, ignore_index=True)
                keywords_sentiment_df = keywords_sentiment_df[['sentiment', 'keyword', 'count']]
                keywords_sentiment_df.to_excel(writer, sheet_name='palabras_por_sentimiento', index=False)

        if not comments_excel.empty:
            trend_df = (
                comments_excel.assign(comment_date=pd.to_datetime(comments_excel['comment_published_at']).dt.date)
                .groupby(['video_id', 'comment_date'])['polarity']
                .mean()
                .reset_index(name='avg_polarity')
            )
            trend_df.to_excel(writer, sheet_name='tendencia_polaridad', index=False)

    if comments_df.empty:
        return

    sentiment_order = ['Positivo', 'Neutro', 'Negativo']
    sentiment_counts = comments_df['sentiment'].value_counts().reindex(sentiment_order, fill_value=0)

    plt.figure(figsize=(9, 5))
    sentiment_counts.plot(kind='bar', color=['#4CAF50', '#9E9E9E', '#F44336'])
    if len(summary_df) == 1:
        plot_title = f"Mineria de Opiniones: {summary_df.iloc[0]['video_title']}"
    else:
        plot_title = f"Mineria de Opiniones: {len(summary_df)} videos"
    plt.title(plot_title)
    plt.xlabel('Categoria de Sentimiento')
    plt.ylabel('Numero de Comentarios')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=140)
    plt.close()


def _pdf_table(ax, data_dict, col_widths=(0.45, 0.55)):
    ax.axis('off')
    rows = list(data_dict.items())
    table = ax.table(
        cellText=rows,
        colLabels=['Metrica', 'Valor'],
        cellLoc='left',
        loc='center',
        colWidths=list(col_widths),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1976D2')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#E3F2FD')
        else:
            cell.set_facecolor('#FFFFFF')
        cell.set_edgecolor('#BDBDBD')


def generate_pdf_report(summary_df, all_comments_df, pdf_path):
    sentiment_colors = {'Positivo': '#4CAF50', 'Neutro': '#9E9E9E', 'Negativo': '#F44336'}

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        fig.patch.set_facecolor('#1565C0')
        ax.set_facecolor('#1565C0')
        ax.text(0.5, 0.72, 'Reporte', ha='center', va='center',
                fontsize=28, fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.5, 0.60, 'Mineria de Datos y Mineria de Opiniones', ha='center',
                va='center', fontsize=18, color='#BBDEFB', transform=ax.transAxes)
        ax.text(0.5, 0.48, 'Plataforma: YouTube', ha='center', va='center',
                fontsize=14, color='#E3F2FD', transform=ax.transAxes)
        ax.text(0.5, 0.38, f'Videos analizados: {len(summary_df)}', ha='center',
                va='center', fontsize=13, color='#E3F2FD', transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        for _, s in summary_df.iterrows():
            vid = s['video_id']
            video_comments = all_comments_df[all_comments_df['video_id'] == vid] if not all_comments_df.empty else pd.DataFrame()

            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor('#FAFAFA')

            header_ax = fig.add_axes([0, 0.88, 1, 0.12])
            header_ax.axis('off')
            header_ax.set_facecolor('#E3F2FD')
            header_ax.text(0.5, 0.65, str(s['video_title'])[:90],
                        ha='center', va='center', fontsize=13,
                        fontweight='bold', color='#0D47A1',
                        transform=header_ax.transAxes)
            header_ax.text(0.5, 0.20, f"Canal: {s['video_channel_title']}  |  ID: {vid}",
                        ha='center', va='center', fontsize=9,
                        color='#0D47A1', transform=header_ax.transAxes)

            table_ax = fig.add_axes([0.03, 0.36, 0.48, 0.50])
            metrics = {
                'Vistas':             f"{int(s['video_view_count']):,}",
                'Likes':              f"{int(s['video_like_count']):,}",
                'Comentarios YouTube': f"{int(s['video_comment_count']):,}",
                'Comentarios analizados': f"{int(s['comments_collected']):,}",
                'Engagement rate':    f"{s['engagement_rate']:.4f}",
                'Polaridad promedio': f"{s['avg_comment_polarity']:.4f}",
                'Polaridad ponderada': f"{s['weighted_comment_polarity']:.4f}",
                'Indice satisfaccion': f"{s['satisfaction_index']:.4f}",
                'Positivos':          f"{int(s['positive_comments'])}  ({s['positive_comments_pct']:.1f}%)",
                'Neutros':            f"{int(s['neutral_comments'])}  ({s['neutral_comments_pct']:.1f}%)",
                'Negativos':          f"{int(s['negative_comments'])}  ({s['negative_comments_pct']:.1f}%)",
            }
            _pdf_table(table_ax, metrics)

            # Grafica de sentimiento
            bar_ax = fig.add_axes([0.55, 0.42, 0.41, 0.42])
            sentiment_order = ['Positivo', 'Neutro', 'Negativo']
            pcts = [
                s['positive_comments_pct'],
                s['neutral_comments_pct'],
                s['negative_comments_pct'],
            ]
            colors = [sentiment_colors[k] for k in sentiment_order]
            bars = bar_ax.bar(sentiment_order, pcts, color=colors, edgecolor='white', linewidth=1.5)
            for bar, pct in zip(bars, pcts):
                bar_ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.8,
                            f'{pct:.1f}%', ha='center', va='bottom',
                            fontsize=10, fontweight='bold')
            bar_ax.set_ylabel('% de comentarios', fontsize=9)
            bar_ax.set_title('Distribucion de Sentimiento', fontsize=10, fontweight='bold')
            bar_ax.set_ylim(0, max(pcts) * 1.25 + 5)
            bar_ax.spines[['top', 'right']].set_visible(False)
            bar_ax.set_facecolor('#F5F5F5')

            kw_ax = fig.add_axes([0.03, 0.06, 0.93, 0.25])
            kw_ax.axis('off')
            kw_ax.text(0, 0.92, 'Palabras clave detectadas:', fontsize=10,
                    fontweight='bold', transform=kw_ax.transAxes, color='#1565C0')
            kw_ax.text(0, 0.65, str(s.get('top_keywords', '')),
                    fontsize=9, transform=kw_ax.transAxes,
                    wrap=True, color='#333333')

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if len(summary_df) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
            fig.patch.set_facecolor('#FAFAFA')
            fig.suptitle('Comparativo entre Videos', fontsize=15,
                        fontweight='bold', color='#1565C0', y=0.97)

            titles = [str(t)[:30] + '...' if len(str(t)) > 30 else str(t)
                    for t in summary_df['video_title']]

            ax1 = axes[0]
            x = range(len(titles))
            ax1.barh(list(x), summary_df['satisfaction_index'].tolist(), color='#1976D2')
            ax1.set_yticks(list(x))
            ax1.set_yticklabels(titles, fontsize=8)
            ax1.set_xlabel('Indice de satisfaccion', fontsize=9)
            ax1.set_title('Satisfaccion por video', fontsize=10, fontweight='bold')
            ax1.spines[['top', 'right']].set_visible(False)
            ax1.set_facecolor('#F5F5F5')

            ax2 = axes[1]
            ax2.barh(list(x), summary_df['engagement_rate'].tolist(), color='#FF7043')
            ax2.set_yticks(list(x))
            ax2.set_yticklabels(titles, fontsize=8)
            ax2.set_xlabel('Engagement rate', fontsize=9)
            ax2.set_title('Engagement por video', fontsize=10, fontweight='bold')
            ax2.spines[['top', 'right']].set_visible(False)
            ax2.set_facecolor('#F5F5F5')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def save_to_sqlite(db_path, summaries_df, comments_df):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS video_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT,
            video_id TEXT,
            video_title TEXT,
            video_channel_id TEXT,
            video_channel_title TEXT,
            video_published_at TEXT,
            video_view_count INTEGER,
            video_like_count INTEGER,
            video_comment_count INTEGER,
            comments_collected INTEGER,
            engagement_rate REAL,
            like_view_ratio REAL,
            comment_view_ratio REAL,
            avg_comment_polarity REAL,
            weighted_comment_polarity REAL,
            positive_comments INTEGER,
            neutral_comments INTEGER,
            negative_comments INTEGER
        )
        '''
    )

    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            comment_id TEXT,
            video_id TEXT,
            author_name TEXT,
            author_channel_id TEXT,
            comment_text TEXT,
            comment_like_count INTEGER,
            comment_published_at TEXT,
            comment_updated_at TEXT,
            reply_count INTEGER,
            polarity REAL,
            sentiment TEXT,
            FOREIGN KEY (run_id) REFERENCES video_runs(run_id)
        )
        '''
    )

    for _, summary in summaries_df.iterrows():
        cur.execute(
            '''
            INSERT INTO video_runs (
                run_at, video_id, video_title, video_channel_id, video_channel_title, video_published_at,
                video_view_count, video_like_count, video_comment_count, comments_collected,
                engagement_rate, like_view_ratio, comment_view_ratio, avg_comment_polarity,
                weighted_comment_polarity, positive_comments, neutral_comments, negative_comments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                summary['run_at'],
                summary['video_id'],
                summary['video_title'],
                summary['video_channel_id'],
                summary['video_channel_title'],
                summary['video_published_at'],
                int(summary['video_view_count']),
                int(summary['video_like_count']),
                int(summary['video_comment_count']),
                int(summary['comments_collected']),
                float(summary['engagement_rate']),
                float(summary['like_view_ratio']),
                float(summary['comment_view_ratio']),
                float(summary['avg_comment_polarity']),
                float(summary['weighted_comment_polarity']),
                int(summary['positive_comments']),
                int(summary['neutral_comments']),
                int(summary['negative_comments']),
            )
        )

        run_id = cur.lastrowid
        current_video_comments = comments_df[comments_df['video_id'] == summary['video_id']]

        if not current_video_comments.empty:
            rows = [
                (
                    run_id,
                    row['comment_id'],
                    row['video_id'],
                    row['author_name'],
                    row['author_channel_id'],
                    row['comment_text'],
                    int(row['comment_like_count']),
                    row['comment_published_at'],
                    row['comment_updated_at'],
                    int(row['reply_count']),
                    float(row['polarity']),
                    row['sentiment'],
                )
                for _, row in current_video_comments.iterrows()
            ]

            cur.executemany(
                '''
                INSERT INTO comments (
                    run_id, comment_id, video_id, author_name, author_channel_id, comment_text,
                    comment_like_count, comment_published_at, comment_updated_at, reply_count,
                    polarity, sentiment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                rows,
            )

    conn.commit()
    conn.close()


def main():
    args = parse_args()
    max_comments = min(max(args.max_comments, 1), MAX_COMMENTS_ALLOWED)
    video_ids = normalize_video_ids(args)

    client_secret_file = resolve_client_secret_file()
    service = Create_Service(client_secret_file, API_NAME, API_VERSION, SCOPES)
    if service is None:
        raise RuntimeError('No fue posible crear el servicio de YouTube.')

    comments_frames = []

    for video_id in video_ids:
        try:
            video_item = get_video_details(service, video_id)
            channel_item = get_channel_details(service, video_item['snippet']['channelId'])
            comments = fetch_comments(service, video_id, max_comments)
            comments_df = enrich_comment_features(pd.DataFrame(comments))
            comments_df['video_title'] = video_item['snippet'].get('title')
            comments_df['video_channel_title'] = video_item['snippet'].get('channelTitle')
            comments_df['video_channel_id'] = video_item['snippet'].get('channelId')
            comments_frames.append(
                {
                    'video_id': video_id,
                    'video_item': video_item,
                    'channel_item': channel_item,
                    'comments_df': comments_df,
                }
            )
        except Exception as error:  # pylint: disable=broad-except
            print(f"Advertencia: no se pudo procesar el video {video_id}: {error}")

    if not comments_frames:
        raise RuntimeError('No se pudo procesar ningun video. Revisa los IDs y vuelve a intentar.')

    all_comments_df = pd.concat([entry['comments_df'] for entry in comments_frames], ignore_index=True)
    all_comments_df = apply_sentiment_model(all_comments_df, args)

    summaries = []
    for entry in comments_frames:
        video_comments_df = all_comments_df[all_comments_df['video_id'] == entry['video_id']].copy()
        summary = build_summary(entry['video_item'], entry['channel_item'], video_comments_df)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    save_outputs(all_comments_df, summary_df, args.excel_path, args.csv_path, args.plot_path)
    save_to_sqlite(args.db_path, summary_df, all_comments_df)
    generate_pdf_report(summary_df, all_comments_df, args.pdf_path)

    for _, summary in summary_df.iterrows():
        print(f"\n--- REPORTE DE MINERIA PARA: {summary['video_title']} ---")
        print(f"Canal: {summary['video_channel_title']}")
        print(f"Vistas: {summary['video_view_count']}")
        print(f"Likes: {summary['video_like_count']}")
        print(f"Comentarios totales en YouTube: {summary['video_comment_count']}")
        print(f"Comentarios descargados (max 100): {summary['comments_collected']}")
        print(f"Engagement rate: {summary['engagement_rate']:.4f}")
        print(f"Polaridad promedio: {summary['avg_comment_polarity']:.4f}")
        print(f"Polaridad ponderada: {summary['weighted_comment_polarity']:.4f}")
        print(
            f"Sentimientos (%): +{summary['positive_comments_pct']:.1f} | "
            f"={summary['neutral_comments_pct']:.1f} | -{summary['negative_comments_pct']:.1f}"
        )
        print(f"Indice de satisfaccion: {summary['satisfaction_index']:.4f}")
        print(f"Palabras clave: {summary['top_keywords']}")

    print(f"\nArchivos generados: {args.excel_path}, {args.csv_path}, {args.plot_path}, {args.db_path}, {args.pdf_path}")


if __name__ == '__main__':
    main()