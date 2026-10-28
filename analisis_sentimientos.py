
import argparse
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

LEXICO_ES = {
    
    'excelente': 3, 'maravilloso': 3, 'increíble': 3, 'fantástico': 3,
    'genial': 3, 'perfecto': 3, 'extraordinario': 3, 'espectacular': 3,
    'feliz': 3, 'felicidad': 3, 'alegría': 3, 'hermoso': 3,
    'brillante': 3, 'outstanding': 3, 'grandioso': 3, 'magnífico': 3,

    'bueno': 2, 'bien': 2, 'bonito': 2, 'amor': 2, 'amar': 2,
    'esperanza': 2, 'mejoró': 2, 'mejorar': 2, 'mejora': 2,
    'salvó': 2, 'salvo': 2, 'salvaste': 2, 'ayudó': 2, 'ayuda': 2,
    'útil': 2, 'beneficio': 2, 'beneficia': 2, 'curación': 2, 'curar': 2,
    'recuperar': 2, 'recuperé': 2, 'recuperó': 2, 'superé': 2, 'superar': 2,
    'tranquilo': 2, 'tranquilidad': 2, 'calma': 2, 'paz': 2,
    'satisfecho': 2, 'satisfacción': 2, 'contento': 2, 'alegre': 2,

    'gracias': 1, 'interesante': 1, 'importante': 1, 'aprendo': 1,
    'aprender': 1, 'entender': 1, 'entendí': 1, 'claro': 1, 'correcto': 1,
    'recomiendo': 1, 'recomendable': 1, 'comparto': 1, 'suscribir': 1,
    'valioso': 1, 'verdad': 1, 'efectivo': 1, 'funciona': 1, 'funcional': 1,

    'horrible': -3, 'terrible': -3, 'devastador': -3, 'insoportable': -3,
    'muero': -3, 'morir': -3, 'muerte': -3, 'matar': -3, 'suicidio': -3,
    'suicidar': -3, 'odio': -3, 'odiar': -3, 'desesperado': -3,
    'desesperanza': -3, 'agonía': -3, 'tortura': -3,

    'triste': -2, 'tristeza': -2, 'llorar': -2, 'lloro': -2, 'llorando': -2,
    'dolor': -2, 'doloroso': -2, 'sufro': -2, 'sufrir': -2, 'sufrimiento': -2,
    'miedo': -2, 'terror': -2, 'angustia': -2, 'angustiado': -2,
    'soledad': -2, 'solo': -2, 'abandonado': -2, 'abandono': -2,
    'perdido': -2, 'perder': -2, 'fracaso': -2, 'fracasar': -2,
    'inútil': -2, 'culpa': -2, 'culpable': -2, 'vergüenza': -2,
    'rechazado': -2, 'rechazo': -2, 'crisis': -2, 'colapso': -2,

    'depresión': -1, 'deprimido': -1, 'ansiedad': -1, 'ansioso': -1,
    'nervioso': -1, 'preocupado': -1, 'preocupación': -1, 'cansado': -1,
    'cansancio': -1, 'agotado': -1, 'agotamiento': -1, 'mal': -1,
    'malo': -1, 'peor': -1, 'difícil': -1, 'dificultad': -1,
    'problema': -1, 'problemas': -1, 'complicado': -1, 'confundido': -1,
    'frustrado': -1, 'frustración': -1, 'enojo': -1, 'enojado': -1,
    'molesto': -1, 'molestar': -1, 'falla': -1, 'fallar': -1,
}

INTENSIFICADORES = {
    'muy': 1.5, 'mucho': 1.3, 'demasiado': 1.4, 'bastante': 1.2,
    'super': 1.5, 'súper': 1.5, 'extremadamente': 1.6, 'totalmente': 1.3,
    'completamente': 1.3, 'absolutamente': 1.3, 'tan': 1.2,
}

NEGADORES = {'no', 'nunca', 'jamás', 'ni', 'sin', 'tampoco'}


def limpiar_texto(texto):
    """Limpia el texto igual que en demo.py."""
    raw = str(texto or '').lower()
    raw = re.sub(r'https?://\S+|www\.\S+', ' ', raw)
    raw = re.sub(r'[@#]\w+', ' ', raw)
    raw = re.sub(r'[^a-z0-9\sáéíóúñü]', ' ', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw


def calcular_score_lexico(texto):
    """
    Calcula un score de sentimiento usando el léxico en español.
    Considera intensificadores y negadores.
    Retorna un float entre -1.0 y 1.0.
    """
    tokens = limpiar_texto(texto).split()
    if not tokens:
        return 0.0

    score_total = 0.0
    intensificador_activo = 1.0
    negador_activo = False

    for i, token in enumerate(tokens):
        if token in NEGADORES:
            negador_activo = True
            continue

        if token in INTENSIFICADORES:
            intensificador_activo = INTENSIFICADORES[token]
            continue

        if token in LEXICO_ES:
            score = LEXICO_ES[token] * intensificador_activo
            if negador_activo:
                score = -score
            score_total += score

        intensificador_activo = 1.0
        negador_activo = False

    max_posible = len(tokens) * 3.0 * 1.6
    if max_posible == 0:
        return 0.0

    normalizado = score_total / max_posible
    return max(-1.0, min(1.0, normalizado))


def clasificar_sentimiento(score, umbral=0.02):
    """
    Clasifica el score en Positivo, Negativo o Neutro.
    El umbral evita que scores muy cercanos a 0 sean mal clasificados.
    """
    if score > umbral:
        return 'Positivo'
    if score < -umbral:
        return 'Negativo'
    return 'Neutro'


STOPWORDS_ES = {
    'a', 'al', 'algo', 'algun', 'alguna', 'algunas', 'alguno', 'algunos',
    'ante', 'antes', 'con', 'contra', 'cual', 'cuando', 'de', 'del',
    'desde', 'donde', 'dos', 'el', 'ella', 'ellas', 'ello', 'ellos',
    'en', 'entre', 'era', 'es', 'esa', 'esas', 'ese', 'eso', 'esos',
    'esta', 'estar', 'este', 'esto', 'estoy', 'fue', 'han', 'hasta',
    'hay', 'la', 'las', 'le', 'les', 'lo', 'los', 'me', 'mi', 'mis',
    'muy', 'ni', 'no', 'nos', 'o', 'otra', 'otros', 'para', 'pero',
    'por', 'que', 'qui', 'se', 'si', 'sin', 'sobre', 'son', 'su', 'sus',
    'te', 'todo', 'todos', 'tu', 'tus', 'un', 'una', 'uno', 'unos',
    'ya', 'yo', 'más', 'así', 'qué', 'cómo', 'dónde', 'cuándo',
}


def extraer_palabras_clave(series, top_n=20):
    tokens = []
    for valor in series.fillna(''):
        for token in str(valor).split():
            if len(token) >= 3 and token not in STOPWORDS_ES:
                tokens.append(token)
    counts = Counter(tokens)
    return pd.DataFrame(counts.most_common(top_n), columns=['keyword', 'count'])


def recalcular_sentimientos(df):
    """Aplica el léxico español a la columna comment_text_clean."""
    print("Recalculando sentimientos con léxico en español...")

    col_texto = 'comment_text_clean' if 'comment_text_clean' in df.columns else 'comment_text'

    df['polarity'] = df[col_texto].fillna('').apply(calcular_score_lexico)
    df['sentiment'] = df['polarity'].apply(clasificar_sentimiento)

    print(f"\nDistribución de sentimientos (nuevo léxico):")
    print(df['sentiment'].value_counts())
    print(f"\nTotal comentarios: {len(df)}")
    return df


def recalcular_resumen(df_comentarios, df_resumen_original):
    """Recalcula el resumen por video con los nuevos sentimientos."""
    resumen_rows = []

    for _, row_resumen in df_resumen_original.iterrows():
        vid = row_resumen['video_id']
        comentarios_video = df_comentarios[df_comentarios['video_id'] == vid]

        n = len(comentarios_video)
        if n == 0:
            resumen_rows.append(row_resumen.to_dict())
            continue

        counts = comentarios_video['sentiment'].value_counts().to_dict()
        pos = int(counts.get('Positivo', 0))
        neu = int(counts.get('Neutro', 0))
        neg = int(counts.get('Negativo', 0))

        nueva_fila = row_resumen.to_dict()
        nueva_fila['avg_comment_polarity'] = float(comentarios_video['polarity'].mean())
        nueva_fila['positive_comments'] = pos
        nueva_fila['neutral_comments'] = neu
        nueva_fila['negative_comments'] = neg
        nueva_fila['positive_comments_pct'] = pos / n * 100
        nueva_fila['neutral_comments_pct'] = neu / n * 100
        nueva_fila['negative_comments_pct'] = neg / n * 100
        nueva_fila['satisfaction_index'] = (pos - neg) / n

        col_texto = 'comment_text_clean' if 'comment_text_clean' in comentarios_video.columns else 'comment_text'
        kw_df = extraer_palabras_clave(comentarios_video[col_texto], top_n=8)
        nueva_fila['top_keywords'] = ', '.join(kw_df['keyword'].tolist())

        resumen_rows.append(nueva_fila)

    return pd.DataFrame(resumen_rows)


def generar_pdf(df_resumen, df_comentarios, pdf_path):
    """Genera el PDF ejecutivo con los resultados corregidos."""
    print(f"\nGenerando PDF en {pdf_path}...")
    colores = {'Positivo': '#4CAF50', 'Neutro': '#9E9E9E', 'Negativo': '#F44336'}

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        fig.patch.set_facecolor('#1565C0')
        ax.set_facecolor('#1565C0')
        ax.text(0.5, 0.72, 'Reporte Ejecutivo', ha='center', fontsize=28,
                fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.5, 0.60, 'Minería de Datos y Minería de Opiniones', ha='center',
                fontsize=18, color='#BBDEFB', transform=ax.transAxes)
        ax.text(0.5, 0.48, 'Plataforma: YouTube', ha='center',
                fontsize=14, color='#E3F2FD', transform=ax.transAxes)
        ax.text(0.5, 0.38, f'Videos analizados: {len(df_resumen)}', ha='center',
                fontsize=13, color='#E3F2FD', transform=ax.transAxes)
        ax.text(0.5, 0.28, 'Modelo: Léxico en Español (corregido)', ha='center',
                fontsize=11, color='#90CAF9', transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        for _, s in df_resumen.iterrows():
            vid = s['video_id']
            coms = df_comentarios[df_comentarios['video_id'] == vid] if not df_comentarios.empty else pd.DataFrame()

            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor('#FAFAFA')

            header_ax = fig.add_axes([0, 0.88, 1, 0.12])
            header_ax.axis('off')
            header_ax.set_facecolor('#E3F2FD')
            titulo = str(s.get('video_title', vid))[:90]
            header_ax.text(0.5, 0.65, titulo, ha='center', fontsize=13,
                        fontweight='bold', color='#0D47A1', transform=header_ax.transAxes)
            canal = str(s.get('video_channel_title', ''))
            header_ax.text(0.5, 0.20, f"Canal: {canal}  |  ID: {vid}",
                        ha='center', fontsize=9, color='#0D47A1', transform=header_ax.transAxes)

            table_ax = fig.add_axes([0.03, 0.36, 0.48, 0.50])
            table_ax.axis('off')
            metricas = {
                'Vistas':              f"{int(s.get('video_view_count', 0)):,}",
                'Likes':               f"{int(s.get('video_like_count', 0)):,}",
                'Comentarios YouTube': f"{int(s.get('video_comment_count', 0)):,}",
                'Comentarios analizados': f"{int(s.get('comments_collected', 0)):,}",
                'Engagement rate':     f"{s.get('engagement_rate', 0):.4f}",
                'Polaridad promedio':  f"{s.get('avg_comment_polarity', 0):.4f}",
                'Índice satisfacción': f"{s.get('satisfaction_index', 0):.4f}",
                'Positivos':           f"{int(s.get('positive_comments', 0))}  ({s.get('positive_comments_pct', 0):.1f}%)",
                'Neutros':             f"{int(s.get('neutral_comments', 0))}  ({s.get('neutral_comments_pct', 0):.1f}%)",
                'Negativos':           f"{int(s.get('negative_comments', 0))}  ({s.get('negative_comments_pct', 0):.1f}%)",
            }
            rows = list(metricas.items())
            table = table_ax.table(cellText=rows, colLabels=['Métrica', 'Valor'],
                                cellLoc='left', loc='center', colWidths=[0.45, 0.55])
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

            bar_ax = fig.add_axes([0.55, 0.42, 0.41, 0.42])
            orden = ['Positivo', 'Neutro', 'Negativo']
            pcts = [s.get('positive_comments_pct', 0),
                    s.get('neutral_comments_pct', 0),
                    s.get('negative_comments_pct', 0)]
            bars = bar_ax.bar(orden, pcts,
                            color=[colores[k] for k in orden],
                            edgecolor='white', linewidth=1.5)
            for bar, pct in zip(bars, pcts):
                bar_ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.8,
                            f'{pct:.1f}%', ha='center', va='bottom',
                            fontsize=10, fontweight='bold')
            bar_ax.set_ylabel('% de comentarios', fontsize=9)
            bar_ax.set_title('Distribución de Sentimiento', fontsize=10, fontweight='bold')
            bar_ax.set_ylim(0, max(pcts) * 1.25 + 5)
            bar_ax.spines[['top', 'right']].set_visible(False)
            bar_ax.set_facecolor('#F5F5F5')

            kw_ax = fig.add_axes([0.03, 0.06, 0.93, 0.25])
            kw_ax.axis('off')
            kw_ax.text(0, 0.92, 'Palabras clave detectadas:', fontsize=10,
                    fontweight='bold', color='#1565C0', transform=kw_ax.transAxes)
            kw_ax.text(0, 0.65, str(s.get('top_keywords', '')),
                    fontsize=9, color='#333333', transform=kw_ax.transAxes)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if len(df_resumen) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
            fig.patch.set_facecolor('#FAFAFA')
            fig.suptitle('Comparativo entre Videos', fontsize=15,
                        fontweight='bold', color='#1565C0', y=0.97)
            titles = [str(t)[:30] + '...' if len(str(t)) > 30 else str(t)
                    for t in df_resumen['video_title']]
            x = range(len(titles))

            axes[0].barh(list(x), df_resumen['satisfaction_index'].tolist(), color='#1976D2')
            axes[0].set_yticks(list(x))
            axes[0].set_yticklabels(titles, fontsize=8)
            axes[0].set_xlabel('Índice de satisfacción', fontsize=9)
            axes[0].set_title('Satisfacción por video', fontsize=10, fontweight='bold')
            axes[0].spines[['top', 'right']].set_visible(False)
            axes[0].set_facecolor('#F5F5F5')

            axes[1].barh(list(x), df_resumen['engagement_rate'].tolist(), color='#FF7043')
            axes[1].set_yticks(list(x))
            axes[1].set_yticklabels(titles, fontsize=8)
            axes[1].set_xlabel('Engagement rate', fontsize=9)
            axes[1].set_title('Engagement por video', fontsize=10, fontweight='bold')
            axes[1].spines[['top', 'right']].set_visible(False)
            axes[1].set_facecolor('#F5F5F5')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"PDF generado: {pdf_path}")


def guardar_excel(df_comentarios, df_resumen, excel_path):
    """Guarda los resultados actualizados en un nuevo Excel."""
    print(f"\nGuardando Excel en {excel_path}...")

    comparativo_cols = [
        'video_id', 'video_title', 'comments_collected', 'engagement_rate',
        'avg_comment_polarity', 'positive_comments_pct', 'neutral_comments_pct',
        'negative_comments_pct', 'satisfaction_index', 'top_keywords',
    ]
    comparativo_df = df_resumen[[c for c in comparativo_cols if c in df_resumen.columns]].sort_values(
        by=['satisfaction_index', 'engagement_rate'], ascending=[False, False]
    )

    col_texto = 'comment_text_clean' if 'comment_text_clean' in df_comentarios.columns else 'comment_text'
    kw_global = extraer_palabras_clave(df_comentarios[col_texto], top_n=30)

    kw_por_sentimiento = []
    for sent in ['Positivo', 'Neutro', 'Negativo']:
        subset = df_comentarios[df_comentarios['sentiment'] == sent]
        parcial = extraer_palabras_clave(subset[col_texto], top_n=15)
        if not parcial.empty:
            parcial['sentiment'] = sent
            kw_por_sentimiento.append(parcial)
    kw_sent_df = pd.concat(kw_por_sentimiento, ignore_index=True)[['sentiment', 'keyword', 'count']] if kw_por_sentimiento else pd.DataFrame()

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_comentarios.to_excel(writer, sheet_name='comentarios', index=False)
        df_resumen.to_excel(writer, sheet_name='resumen', index=False)
        comparativo_df.to_excel(writer, sheet_name='comparativo_videos', index=False)
        kw_global.to_excel(writer, sheet_name='palabras_clave', index=False)
        if not kw_sent_df.empty:
            kw_sent_df.to_excel(writer, sheet_name='palabras_por_sentimiento', index=False)

    print(f"Excel guardado: {excel_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Recalcula sentimientos con léxico en español.')
    parser.add_argument('--input',  default='mineria_buap_final.xlsx', help='Excel de entrada')
    parser.add_argument('--output', default='mineria_corregida.xlsx',  help='Excel de salida')
    parser.add_argument('--pdf',    default='reporte_corregido.pdf',   help='PDF de salida')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Leyendo {args.input}...")
    df_comentarios = pd.read_excel(args.input, sheet_name='comentarios')
    df_resumen_orig = pd.read_excel(args.input, sheet_name='resumen')

    print(f"\nDistribución ORIGINAL (TextBlob):")
    print(df_comentarios['sentiment'].value_counts())

    df_comentarios = recalcular_sentimientos(df_comentarios)
    df_resumen_nuevo = recalcular_resumen(df_comentarios, df_resumen_orig)

    guardar_excel(df_comentarios, df_resumen_nuevo, args.output)
    generar_pdf(df_resumen_nuevo, df_comentarios, args.pdf)

    print("\n✓ Proceso completado.")
    print(f"  Excel:  {args.output}")
    print(f"  PDF:    {args.pdf}")

if __name__ == '__main__':
    main()
