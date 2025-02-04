# databricks_download.py
import requests

def download_satellite_image():
    # Konfiguration: Setze deinen Token und die Instanz-URL
    token = "YOUR_DATABRICKS_TOKEN"  # Ersetze durch deinen persönlichen Token
    databricks_instance = "https://<databricks-instance>"  # z. B. "https://adb-1234567890123456.7.azuredatabricks.net"

    # API-Endpunkt zum Lesen einer Datei aus DBFS
    endpoint = f"{databricks_instance}/api/2.0/dbfs/read"

    # Der Pfad zur Datei im DBFS
    file_path = "/FileStore/satellite_images/sample.jpg"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    # Anfrageparameter: Der API-Aufruf erwartet den Pfad als Parameter
    params = {
        "path": file_path
    }

    response = requests.get(endpoint, headers=headers, params=params)

    if response.status_code == 200:
        # Speichere das Bild lokal ab
        with open("sample.jpg", "wb") as f:
            f.write(response.content)
        print("Bild erfolgreich heruntergeladen und als 'sample.jpg' gespeichert.")
        return "sample.jpg"
    else:
        error_msg = f"Fehler beim Herunterladen des Bildes: {response.status_code} - {response.text}"
        raise Exception(error_msg)

if __name__ == "__main__":
    download_satellite_image()
