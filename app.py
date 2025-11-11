# Importy bibliotek
import streamlit as st
import boto3
import tempfile
from dotenv import load_dotenv, dotenv_values
import os  
import pandas as pd
from pycaret.regression import load_model, predict_model
import numpy as np
import matplotlib.pyplot as plt
import json
from langfuse.decorators import observe
from langfuse.openai import OpenAI

# Wczytanie zmiennych środowiskowych
load_dotenv()  # Wczytuje .env jeśli istnieje (lokalnie)

# Konfiguracja Langfuse - używa os.environ (działa lokalnie i na Digital Ocean)
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Konfiguracja strony
st.set_page_config(
    page_title="Prognoza Półmaratonu 🔮",
    page_icon="🔮",
    layout="wide"
)

# Nagłówek z informacją
st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; 
                border-radius: 15px; 
                text-align: center;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                margin-bottom: 40px;'>
        <h1 style='color: white; margin: 0; font-size: 3em;'>🔮</h1>
        <h2 style='color: white; margin-top: 20px; line-height: 1.6;'>
            Sportowa wróżka przewidzi czas, jaki potrzebujesz, aby przebiec półmaraton.
        </h2>
        <p style='color: #f0f0f0; font-size: 18px; margin-top: 20px;'>
            Wpisz poniżej podstawowe informacje o sobie, takie jak <strong>płeć</strong>, 
            <strong>wiek</strong>, <strong>czas, w jakim przebiegasz 5 kilometrów</strong>.
        </p>
    </div>
""", unsafe_allow_html=True)

# Pole tekstowe
user_input = st.text_area(
    label="Twoje dane:",
    placeholder="Przykład: Jestem mężczyzną, mam 32 lata i przebiegam 5 km w 24 minuty i 15 sekund.",
    height=150,
    help="Wpisz informacje o swojej płci, wieku i czasie biegu na 5 km"
)

# Inicjalizacja OpenAI z Langfuse
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ładujemy model wytrenowany
# Konfiguracja Digital Ocean Spaces (kompatybilny z S3 API)
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "https://fra1.digitaloceanspaces.com"),  # Frankfurt endpoint
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),  # DO Spaces Access Key (DO00BZ...)
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),  # DO Spaces Secret Key
    region_name="fra1"  # Digital Ocean Frankfurt region
)
BUCKET_NAME = "pracadomowamodul9"
with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
    s3.download_file(BUCKET_NAME, 'processed/best_model.pkl', tmp_file.name)
    temp_path = tmp_file.name

    from pycaret.regression import load_model
best_final_model = load_model(temp_path.replace('.pkl', ''))
os.unlink(temp_path)


# ============== ETAP 2: ANALIZA DANYCH PRZEZ AI ==============

# Funkcja z obserwacją Langfuse
@observe(name="extract_runner_data")
def extract_runner_data(user_text):
    """
    Wyodrębnia dane biegacza z tekstu użytkownika przy użyciu OpenAI.
    Monitorowane przez Langfuse.
    
    Args:
        user_text: Tekst wprowadzony przez użytkownika
        
    Returns:
        dict: Słownik z wyodrębnionymi danymi (plec, wiek, czas_5km_sekundy)
    """
    system_prompt = """Jesteś asystentem analizującym dane biegaczy. 
    Z podanego tekstu wyodrębnij następujące informacje:
    1. Płeć: 'M' dla mężczyzny lub 'K' dla kobiety
    2. Wiek: liczba całkowita (w latach)
    3. Czas biegu na 5 km w sekundach: przelicz podany czas (minuty/sekundy) na całkowitą liczbę sekund
    
    Zwróć odpowiedź TYLKO w formacie JSON bez dodatkowych komentarzy:
    {
        "plec": "M" lub "K",
        "wiek": liczba,
        "czas_5km_sekundy": liczba
    }
    
    Jeśli jakiejś informacji nie można określić, użyj null.
    """
    
    user_prompt = f"Tekst użytkownika: {user_text}"
    
    # Wywołanie OpenAI API (automatycznie śledzone przez Langfuse)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    
    # Parsowanie odpowiedzi JSON
    ai_response = response.choices[0].message.content
    extracted_data = json.loads(ai_response)
    
    return extracted_data


# Przycisk do analizy danych
if st.button("🔮 Analizuj dane", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("🤖 Wróżka analizuje Twoje dane..."):
            
            try:
                # Wywołanie funkcji z Langfuse tracking
                extracted_data = extract_runner_data(user_input)
                
                # Pobieranie danych
                plec = extracted_data.get("plec")
                wiek = extracted_data.get("wiek")
                czas_5km = extracted_data.get("czas_5km_sekundy")
                
                # Walidacja danych
                missing_fields = []
                if not plec:
                    missing_fields.append("płeć (mężczyzna/kobieta)")
                if not wiek:
                    missing_fields.append("wiek")
                if not czas_5km:
                    missing_fields.append("czas biegu na 5 km")
                
                if missing_fields:
                    st.error(f"❌ Brakujące informacje: {', '.join(missing_fields)}. Proszę uzupełnić dane.")
                else:
                    # Określenie kategorii wiekowej
                    def get_kategoria_wiekowa(plec, wiek):
                        """Określa kategorię wiekową na podstawie płci i wieku"""
                        if 21 <= wiek <= 30:
                            return f"{plec}20"
                        elif 31 <= wiek <= 40:
                            return f"{plec}30"
                        elif 41 <= wiek <= 50:
                            return f"{plec}40"
                        elif 51 <= wiek <= 60:
                            return f"{plec}50"
                        elif 61 <= wiek <= 70:
                            return f"{plec}60"
                        elif 71 <= wiek <= 80:
                            return f"{plec}70"
                        elif 81 <= wiek <= 90:
                            return f"{plec}80"
                        else:
                            return None
                    
                    kategoria = get_kategoria_wiekowa(plec, wiek)
                    
                    if not kategoria:
                        st.error(f"❌ Wiek {wiek} lat nie mieści się w obsługiwanych kategoriach (21-90 lat).")
                    else:
                        # Utworzenie DataFrame dla predykcji
                        predict_df = pd.DataFrame([
                            {
                                "Płeć": plec, 
                                "Kategoria wiekowa": kategoria, 
                                "5 km Czas": czas_5km,
                            }
                        ])
                        
                        # Dodanie obliczonych kolumn
                        predict_df['5 km Tempo'] = (predict_df['5 km Czas'] / 60) / 5
                        predict_df['Tempo Stabilność'] = 0.0415  
                        predict_df['Tempo'] = predict_df['5 km Tempo']
                        
                        # Zapisanie DataFrame w session_state dla kolejnych etapów
                        st.session_state['predict_df'] = predict_df
                        st.session_state['plec'] = plec
                        st.session_state['kategoria'] = kategoria
                        st.session_state['czas_5km'] = czas_5km
                        
                        # Sukces!
                        st.success("✅ Dane przeanalizowane pomyślnie!")
                        
                        # Wyświetlenie informacji weryfikacyjnych
                        st.markdown("---")
                        st.markdown("### 📋 Potwierdzenie Twoich danych:")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                label="Twoja płeć",
                                value="Mężczyzna" if plec == "M" else "Kobieta"
                            )
                        
                        with col2:
                            st.metric(
                                label="Twoja kategoria wiekowa",
                                value=kategoria
                            )
                        
                        with col3:
                            minuty = czas_5km // 60
                            sekundy = czas_5km % 60
                            st.metric(
                                label="Twój czas (5km)",
                                value=f"{minuty}:{sekundy:02d} ({czas_5km}s)"
                            )
                        
                        # ============== ETAP 3: PREDYKCJA CZASU ==============
                        st.markdown("---")
                        
                        with st.spinner("🔮 Wróżka przewiduje Twój czas..."):
                            # Predykcja za pomocą modelu
                            predict_czas = predict_model(best_final_model, data=predict_df)
                            
                            # Pobranie przewidywanego czasu (w sekundach)
                            czas_polmaratonu_sekundy = int(predict_czas['prediction_label'].iloc[0])
                            
                            # Konwersja na godziny:minuty:sekundy
                            godziny = czas_polmaratonu_sekundy // 3600
                            minuty_pozostale = (czas_polmaratonu_sekundy % 3600) // 60
                            sekundy_pozostale = czas_polmaratonu_sekundy % 60
                            
                            # Zapisanie wyniku w session_state
                            st.session_state['czas_polmaratonu'] = czas_polmaratonu_sekundy
                            st.session_state['czas_format'] = f"{godziny}:{minuty_pozostale:02d}:{sekundy_pozostale:02d}"
                        
                        # Wyświetlenie wyniku
                        st.markdown("""
                            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                        padding: 40px; 
                                        border-radius: 15px; 
                                        text-align: center;
                                        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                                        margin-top: 20px;'>
                                <h2 style='color: white; margin-bottom: 20px;'>
                                    🔮 Wróżka przewiduje, że przebiegniesz półmaraton za:
                                </h2>
                                <h1 style='color: white; font-size: 4em; margin: 20px 0;'>
                                    {czas_format}
                                </h1>
                                <p style='color: white; font-size: 18px;'>
                                    ({czas_polmaratonu_sekundy} sekund)
                                </p>
                            </div>
                        """.format(
                            czas_format=st.session_state['czas_format'],
                            czas_polmaratonu_sekundy=czas_polmaratonu_sekundy
                        ), unsafe_allow_html=True)
                        
                        # ============== ETAP 4: WIZUALIZACJE PORÓWNAWCZE ==============
                        st.markdown("---")
                        st.markdown("### 📊 Porównaj swoje wyniki z innymi biegaczami")
                        
                        # Wczytanie danych biegaczy
                        @st.cache_data
                        def load_runners_data():
                            """Wczytuje dane wszystkich biegaczy z S3"""
                            df = pd.read_csv(f"s3://{BUCKET_NAME}/processed/df.csv")
                            return df
                        
                        df = load_runners_data()
                        
                        # Filtrowanie danych dla tej samej płci i kategorii wiekowej
                        df_filtered = df[
                            (df['Płeć'] == plec) & 
                            (df['Kategoria wiekowa'] == kategoria)
                        ]
                        
                        if len(df_filtered) > 0:
                            st.info(f"📈 Porównujemy Twoje wyniki z **{len(df_filtered)}** biegaczami w kategorii **{kategoria}**")
                            
                            # Histogram
                            st.markdown("#### 📊 Rozkład czasów w Twojej kategorii")
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Histogram
                            ax.hist(df_filtered['Czas'], bins=30, color='#667eea', alpha=0.7, edgecolor='black')
                            
                            # Czerwona linia - czas użytkownika
                            ax.axvline(
                                czas_polmaratonu_sekundy, 
                                color='red', 
                                linestyle='--', 
                                linewidth=3,
                                label=f'Twój przewidywany czas: {st.session_state["czas_format"]}'
                            )
                            
                            ax.set_xlabel('Czas półmaratonu (sekundy)', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Liczba biegaczy', fontsize=12, fontweight='bold')
                            ax.set_title(f'Rozkład czasów - {kategoria}', fontsize=14, fontweight='bold')
                            ax.legend(fontsize=11)
                            ax.grid(True, alpha=0.3)
                            
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.warning(f"⚠️ Brak danych dla kategorii **{kategoria}** w bazie.")
            
            except json.JSONDecodeError:
                st.error("❌ Błąd parsowania odpowiedzi AI. Spróbuj ponownie lub zmień format danych.")
            except Exception as e:
                st.error(f"❌ Wystąpił błąd: {str(e)}")
    else:
        st.warning("⚠️ Proszę wprowadzić swoje dane przed analizą.")

