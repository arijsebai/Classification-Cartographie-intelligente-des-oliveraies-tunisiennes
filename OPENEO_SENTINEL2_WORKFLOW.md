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

Pour l'entrainement segmentation U-Net, il faut des pixels de fond autour de la parcelle. Utiliser donc des exports bbox avec buffer :

```powershell
python scripts\download_sentinel2_openeo.py --start-jobs --export-region bbox --buffer-deg 0.01
```

Pour tester l'elargissement sans lancer 49 jobs, commencer par le split test uniquement :

```powershell
python scripts\download_sentinel2_openeo.py --splits test --start-jobs --export-region bbox --buffer-deg 0.01 --auth-method device --auth-timeout 1800 --no-browser --max-started-jobs 7
```

Cela ecrit :

```text
openeo_jobs_manifest_bbox_test.json
```

Cela ecrit un manifeste separe :

```text
openeo_jobs_manifest_bbox.json
```

Copernicus limite souvent les jobs concurrents a 30. Si certains jobs restent crees mais non demarres, attendre que les premiers finissent puis lancer :

```powershell
python scripts\start_pending_openeo_jobs.py --manifest openeo_jobs_manifest_bbox.json --auth-timeout 1800 --no-browser --max-start 30
```

Si la limite reste bloquee alors que les anciens jobs sont `finished`, telecharger les resultats puis supprimer les anciens jobs du backend :

```powershell
python scripts\download_openeo_results.py --manifest openeo_jobs_manifest.json --output sentinel2_l2a/2025_05_06 --auth-method device --auth-timeout 1800 --no-browser
python scripts\delete_openeo_jobs.py --manifest openeo_jobs_manifest.json --auth-timeout 1800 --no-browser
python scripts\delete_openeo_jobs.py --manifest openeo_jobs_manifest.json --auth-timeout 1800 --no-browser --yes
```

La deuxieme commande sans `--yes` est un apercu. La troisieme supprime reellement les jobs distants termines.

Ces nouveaux GeoTIFF doivent ensuite etre utilises avec `--mask-mode rasterize` cote entrainement/evaluation.

Telecharger les resultats bbox dans un dossier separe :

```powershell
python scripts\download_openeo_results.py --manifest openeo_jobs_manifest_bbox.json --output sentinel2_l2a/2025_05_06_bboxbuf001 --auth-method device --auth-timeout 1800 --no-browser
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

## 5b. Polygones dessines dans la demo

Quand la demo affiche une demande comme `demo_...`, elle a seulement enregistre le polygone dans :

```text
demo_app/submissions/
```

Pour obtenir la vraie classification Sentinel-2 de ces polygones, exporter d'abord les demandes en GeoJSON :

```powershell
python scripts\export_demo_submissions_geojson.py
```

Pour exporter une seule demande :

```powershell
python scripts\export_demo_submissions_geojson.py --job-id demo_1777742534_27532 --output demo_app/submissions/demo_1777742534_27532.geojson
```

Puis creer et demarrer les jobs openEO a partir de ce GeoJSON :

```powershell
python scripts\download_sentinel2_openeo.py --input-geojson demo_app/submissions/submissions.geojson --manifest openeo_jobs_manifest_demo.json --start-jobs --auth-method device --auth-timeout 1800 --no-browser
```

Ou pour une seule demande exportee :

```powershell
python scripts\download_sentinel2_openeo.py --input-geojson demo_app/submissions/demo_1777742534_27532.geojson --manifest openeo_jobs_manifest_demo_1777742534_27532.json --start-jobs --auth-method device --auth-timeout 1800 --no-browser
```

Quand les jobs sont `finished`, telecharger les GeoTIFF :

```powershell
python scripts\download_openeo_results.py --manifest openeo_jobs_manifest_demo.json --output sentinel2_l2a/2025_05_06 --auth-method device --auth-timeout 1800 --no-browser
```

Ensuite, dans la demo, clique de nouveau sur `Analyse automatique` pour afficher la vraie classification Sentinel-2.

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
