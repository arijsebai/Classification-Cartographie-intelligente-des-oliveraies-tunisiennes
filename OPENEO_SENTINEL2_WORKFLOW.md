# Sentinel-2 L2A avec Copernicus openEO

Google Earth Engine n'est pas obligatoire. Cette option utilise Copernicus Data Space Ecosystem avec openEO.

## 1. Installer Python et les dependances

Installer Python depuis https://www.python.org/downloads/ puis, dans le dossier du projet :

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-openeo.txt
```

## 2. Creer un compte Copernicus Data Space

Creer un compte sur Copernicus Data Space Ecosystem, puis lancer l'authentification openEO. Le script ouvre une page de connexion au premier lancement.

## 3. Tester avec une seule parcelle

```powershell
python scripts\download_sentinel2_openeo.py --limit 1
```

Cela cree un job openEO et ecrit :

```text
openeo_jobs_manifest.json
```

Le lien Copernicus s'ouvre normalement dans le navigateur. Connecte-toi avec ton compte Copernicus Data Space avant la fin du delai. Le script attend 15 minutes par defaut.

Si le navigateur ne s'ouvre pas, copie le lien affiche dans le terminal. Pour augmenter le delai :

```powershell
python scripts\download_sentinel2_openeo.py --limit 1 --auth-timeout 1800
```

Si tu te connectes dans le navigateur mais que le terminal reste en `Authorization pending`, ne lance pas `--auth-method auth-code` avec Copernicus Data Space : ce backend peut refuser la redirection locale avec `Invalid parameter: redirect_uri`.

Dans ce cas, relance le mode device avec un nouveau code :

```powershell
python scripts\download_sentinel2_openeo.py --limit 1 --auth-method device --auth-timeout 1800 --no-browser
```

Puis copie-colle manuellement le lien affiche dans le terminal dans ton navigateur. Verifie que la page Copernicus affiche bien une confirmation finale du type autorisation acceptee / device login successful avant de revenir au terminal.

Si le lien avec `?user_code=...` ouvre seulement ton compte mais n'affiche aucun bouton de confirmation, ouvre plutot cette page :

```text
https://identity.dataspace.copernicus.eu/auth/realms/CDSE/device
```

Puis saisis manuellement le code affiche dans le terminal, par exemple :

```text
XDDL-YOFC
```

Cette page doit afficher un champ pour entrer le code device. Apres validation, le terminal doit quitter `Authorization pending`.

## 4. Lancer les jobs

Pour creer et demarrer les jobs pour toutes les parcelles :

```powershell
python scripts\download_sentinel2_openeo.py --start-jobs
```

Parametres par defaut :

```text
collection: SENTINEL2_L2A
periode: 2025-05-01 -> 2025-06-30
bandes: B02, B03, B04, B08, B11
masque SCL: garder 4 vegetation et 5 sol nu
composite: median temporel
sortie: GeoTIFF
```

## 5. Telecharger les resultats termines

Quand les jobs sont finis dans openEO :

```powershell
python scripts\download_openeo_results.py
```

Si les jobs passent en `error`, afficher les logs :

```powershell
python scripts\show_openeo_error_logs.py --auth-timeout 1800 --no-browser
```

Apres correction du script, il faut recreer de nouveaux jobs openEO avec `download_sentinel2_openeo.py`; les anciens jobs en `error` ne peuvent pas etre reutilises.

Les GeoTIFF seront ranges ici :

```text
sentinel2_l2a/
  2025_05_06/
    train/
    val/
    test/
```

## 6. Changer l'annee

Exemple pour 2024 :

```powershell
python scripts\download_sentinel2_openeo.py --year 2024 --start-jobs
```

## Notes

Les scripts utilisent les GeoJSON deja generes dans :

```text
data_splits/ezzayra_oliviers_geojson/
```

Chaque job openEO correspond a une parcelle, ce qui facilite le suivi et evite de melanger les splits.
