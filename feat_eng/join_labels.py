import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

# 1. Настройка окружения для Windows
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] += os.pathsep + "C:\\hadoop\\bin"

spark = SparkSession.builder \
    .appName("join_labels") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .getOrCreate()

train_path = '../datasets/train/'
temp_path = '../datasets/joined'
label_filename = 'train_labels.parquet'
filenames = ['train_part_1.parquet', 'train_part_2.parquet', 'train_part_3.parquet']

if not os.path.exists(temp_path):
    os.makedirs(temp_path)

label_full_path = os.path.join(train_path, label_filename)
if not os.path.exists(label_full_path):
    print(f'Критическая ошибка: {label_filename} не найден!')
    exit(1)

# Используем broadcast, так как файл лейблов обычно небольшой
df_labels = broadcast(spark.read.parquet(label_full_path).select('customer_id', 'event_id', 'target'))

for file in filenames:
    full_path = os.path.normpath(os.path.join(train_path, file))

    if not os.path.exists(full_path):
        print(f'Файл {file} пропущен')
        continue

    print(f'Обработка {file}...')
    df = spark.read.parquet(full_path)
    df_join = df.join(df_labels, on=['customer_id', 'event_id'], how='left')

    # Путь к временной папке, которую создаст Spark
    tmp_folder = os.path.join(temp_path, f"{file}_tmp_dir")
    # Финальный путь к файлу
    final_file_path = os.path.join(temp_path, file)

    # 1. Записываем в одну часть внутри папки
    df_join.coalesce(1).write.mode("overwrite").parquet(tmp_folder)

    # 2. Ищем файл .parquet внутри папки и переносим его наружу
    for f in os.listdir(tmp_folder):
        if f.endswith(".parquet") and not f.startswith("._"):
            shutil.move(os.path.join(tmp_folder, f), final_file_path)
            break
    
    # 3. Удаляем временную папку со всем мусором
    shutil.rmtree(tmp_folder)
    
    print(f'Успешно создан файл: {final_file_path}')

print("Работа завершена.")