import pyspark.sql.types as T
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import SQLContext
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col


spark = SparkSession.builder.appName("CrimeStatistics").getOrCreate()

sc = spark.sparkContext
sqlContext = SQLContext(sc)

crime = spark.read.csv('./crime.csv', header=True)
offense_code = spark.read.csv('./datasets_49781_90476_offense_codes.csv', header=True)


if __name__ == "__main__":

    crime = crime.withColumn("SHOOTING", F.col("SHOOTING").cast(T.IntegerType()))
    crime = crime.withColumn("YEAR", F.col("YEAR").cast(T.IntegerType()))
    crime = crime.withColumn("MONTH", F.col("MONTH").cast(T.IntegerType()))
    crime = crime.withColumn("DAY_OF_WEEK", F.col("DAY_OF_WEEK").cast(T.IntegerType()))
    crime = crime.withColumn("HOUR", F.col("HOUR").cast(T.IntegerType()))
    crime = crime.withColumn("Lat", F.col("Lat").cast(T.FloatType()))
    crime = crime.withColumn("Long", F.col("Long").cast(T.FloatType()))

    # 1. crimes_total - общее количество преступлений в этом районе
    crimes_total = crime.groupBy('district').agg(F.count('district').alias("crimes_total"))

    # 2. crimes_monthly - медиана числа преступлений в месяц в этом районе
    crimes_num = crime.groupBy('month', 'year', 'district').agg(F.count('district').alias("crimes_num"))
    crimes_num.registerTempTable("df")
    crimes_monthly = sqlContext.sql("select district, month, "
                         "percentile_approx(crimes_num, 0.5) as crimes_monthly from df group by month, district "
                         "order by district, month")

    # 3. frequent_crime_types - три самых частых crime_type за всю историю наблюдений в этом районе, объединенных через
    # запятую с одним пробелом “, ” , расположенных в порядке убывания частоты
    offense_code = offense_code.withColumn("crime_type", F.split(F.col("NAME"), " ").getItem(0))
    broadcast_offense_code = F.broadcast(offense_code)
    crime = crime.join(broadcast_offense_code, on=[broadcast_offense_code.CODE == crime.OFFENSE_CODE], how='inner')
    crime_type_count = crime.groupBy('district', "crime_type") \
        .agg(F.count("district").alias('number_of_crime_type_in_district')) \
        .orderBy("number_of_crime_type_in_district", ascending=False)

    window = Window.partitionBy('district', 'crime_type').orderBy(
        crime_type_count["number_of_crime_type_in_district"].desc())
    frequent_crime_types = crime_type_count.select('*', rank().over(window).alias('rank')) \
        .filter(col('rank') <= 3)

    # 4. lat - широта координаты района, расчитанная как среднее по всем широтам инцидентов
    lat = crime.groupBy('district').agg(F.mean('Lat'))

    # 5. lng - широта координаты района, расчитанная как среднее по всем широтам инцидентов
    lng = crime.groupBy('district').agg(F.mean('Long'))