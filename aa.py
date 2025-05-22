import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, data_path):
        """
        İçerik tabanlı öneri sistemi sınıfı
        Args:
            data_path (str): Excel veri seti dosya yolu
        """
        self.df = pd.read_excel(data_path)
        print(f"Veri seti boyutu: {len(self.df)} satır")
        print("Mevcut sütunlar:", self.df.columns.tolist())

        if len(self.df) == 0:
            raise ValueError("Veri seti boş!")

        self.tfidf = TfidfVectorizer(stop_words='english')
        self.cosine_sim = None
        self._prepare_data()

    def _prepare_data(self):
        """Veriyi hazırla ve TF-IDF matrisini oluştur"""
        # Eksik sütunları kontrol et ve oluştur
        required_columns = ['duygu_1', 'duygu_2', 'duygu_3', 'sosyallik', 'zaman_araligi', 'öneri']
        for col in required_columns:
            if col not in self.df.columns:
                print(f"Uyarı: {col} sütunu eksik. Boş sütun oluşturuluyor.")
                self.df[col] = ''

        # NaN değerleri boş string ile değiştir
        self.df = self.df.fillna('')

        # Giriş özelliklerini birleştir
        self.df['combined_features'] = self.df.apply(
            lambda x: f"{x['duygu_1']} {x['duygu_2']} {x['duygu_3']} {x['sosyallik']} {x['zaman_araligi']}",
            axis=1
        )

        # TF-IDF matrisini oluştur
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])

        # Benzerlik matrisini hesapla
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, duygu1, duygu2, duygu3, zaman, sosyallik, top_k=3):
        """
        Verilen girdilere göre öneriler yap
        Args:
            duygu1 (str): Birinci duygu
            duygu2 (str): İkinci duygu
            duygu3 (str): Üçüncü duygu
            zaman (str): Zaman aralığı
            sosyallik (str): Sosyallik seviyesi
            top_k (int): Kaç öneri döndürüleceği
        Returns:
            pd.DataFrame: Öneriler ve benzerlik skorları
        """
        try:
            # Kullanıcı girdilerini birleştir
            user_features = f"{duygu1} {duygu2} {duygu3} {sosyallik} {zaman}"

            # Kullanıcı girdisini TF-IDF vektörüne dönüştür
            user_tfidf = self.tfidf.transform([user_features])

            # Kullanıcı vektörü ile veri setindeki tüm öğeler arasındaki benzerliği hesapla
            cosine_similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()

            # En benzer top_k öğeyi bul
            similar_indices = cosine_similarities.argsort()[::-1][:top_k]
            similar_scores = cosine_similarities[similar_indices]

            # Sonuçları DataFrame olarak döndür
            recommendations = self.df.iloc[similar_indices].copy()
            recommendations['benzerlik_skoru'] = similar_scores

            # Sadece öneri ve benzerlik skoru sütunlarını döndür
            result_columns = ['öneri', 'benzerlik_skoru']

            return recommendations[result_columns]

        except Exception as e:
            print(f"Öneri oluşturulurken hata: {str(e)}")
            return pd.DataFrame(columns=['öneri', 'benzerlik_skoru'])

# Örnek kullanım
if __name__ == "__main__":
    try:
        # Öneri sistemini başlat
        recommender = ContentBasedRecommender('duygu_400.xlsx')

        # Örnek öneri al
        print("\nÖrnek Öneri:")
        recommendations = recommender.recommend(
            duygu1="aşık",
            duygu2="yorgun",
            duygu3="heyecanlı",
            zaman="gece",
            sosyallik="tek",

            top_k=3  # En benzer 3 öneriyi al
        )

        if not recommendations.empty:
            print("\nÖneriler:")
            for idx, row in recommendations.iterrows():
                print(f"Öneri: {row['öneri']}")
                print(f"Benzerlik Skoru: {row['benzerlik_skoru']:.4f}\n")
        else:
            print("\nÖneri bulunamadı!")

    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
        import traceback
        print("\nHata detayları:")
        print(traceback.format_exc())
