import pandas as pd
import pstats

# Charger les statistiques
stats = pstats.Stats("../profile_stats.prof")
stats.strip_dirs().sort_stats("tottime")

# Extraire les données
data = []
for func, (cc, nc, tt, ct, callers) in stats.stats.items():
    for caller in callers:
        data.append([
            pstats.func_std_string(func),  # Fonction lente
            pstats.func_std_string(caller),  # Fonction qui l'appelle
            tt,  # Temps total passé dans la fonction
            ct   # Temps cumulé (avec appels internes)
        ])

# Convertir en DataFrame
df = pd.DataFrame(data, columns=["Fonction lente", "Appelée par", "Temps total (s)", "Temps cumulé (s)"])

# Trier par temps total et afficher les 10 plus longs
df_sorted = df.sort_values(by="Temps total (s)", ascending=False).head(10)
print(df_sorted)
