from pyspark.sql import SparkSession

def main():
    # Erstelle eine SparkSession.
    # Dabei verbindet sich databricks-connect automatisch mit dem Remote-Cluster,
    # sofern die Konfiguration korrekt vorgenommen wurde.
    spark = SparkSession.builder \
        .appName("DatabricksConnectExample") \
        .getOrCreate()

    # Pfad zum Satellitenbild im DBFS (verwende den DBFS-Pfad, z.B. "dbfs:/FileStore/...")
    image_dbfs_path = "dbfs:/FileStore/satellite_images/sample.jpg"

    # Lese die Datei als binäres DataFrame ein. Das Format "binaryFile" ermöglicht es,
    # Dateien (z.B. Bilder) als Binärdaten zu laden.
    df = spark.read.format("binaryFile").load(image_dbfs_path)

    # Zeige das Schema und einige Daten an, um sicherzustellen, dass der Ladevorgang funktioniert.
    df.printSchema()
    df.show(truncate=False)

    # Extrahiere die Binärdaten (falls vorhanden) und speichere das Bild lokal ab.
    # Hier wird angenommen, dass nur eine Datei geladen wurde.
    data = df.collect()
    if data:
        image_data = data[0].content  # 'content' enthält die Binärdaten
        local_image_path = "sample_from_dbfs.jpg"
        with open(local_image_path, "wb") as f:
            f.write(image_data)
        print(f"Bild wurde lokal unter '{local_image_path}' gespeichert.")
    else:
        print("Keine Bilddaten gefunden!")

    spark.stop()

if __name__ == "__main__":
    main()
