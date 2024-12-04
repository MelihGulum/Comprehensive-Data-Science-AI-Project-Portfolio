
# **Real-Time and Batch Data Pipelines with Kafka**

This project focuses on building a real-time and batch data pipeline to process synthetic data (Stock Price and Sales Trend). The pipeline consists of multiple components orchestrated using Docker Compose. The real-time component streams data from a producer to Kafka, and a consumer reads the data from Kafka and stores it in a database (PostgreSQL). The batch processing (ETL) component reads the data from PostgreSQL, calculates average summaries for temperature and humidity over defined intervals, and stores the results back into the database.

The solution utilizes Kafka for real-time data streaming, Python for data processing, and PostgreSQL for data storage and reporting. This project demonstrates how to integrate these technologies into a seamless pipeline with clear modular components, ensuring a scalable and efficient data flow from ingestion to reporting.

Docker Compose is used to manage all services, ensuring that each component (producer, consumer, ETL process, and database) runs in isolated containers, while Kafka acts as the backbone for message streaming.


<p align="center">   
  <img src="https://github.com/user-attachments/assets/91bfcc78-a7b6-46ff-865b-4d49f989d7f4" alt="Diagram "width="100%" height="100%">
</p>
 <p align="center"> <em>Diagram (Image by Author)</em> </p><br>

## **Technologies Used 🛠️**:
* Apache Kafka
* Docker
* PostgreSQL

## **Folder Structure 📂**
```
Real-Time and Batch Data Pipelines with Kafka/
├── consumer/
│   └── consumer.py
├── data/
│   └── synthetic_data.parquet
├── database/
│   └── database.py
├── etl/
│   └── etl.py
├── producer/
│   └── producer.py
├── .env
├── docker-compose.yml
├── Dockerfile
├── init.sql
├── requirements.txt
```


## **Key Points 📌**
### **How to run ▶️**:
* Navigate to the project folder and execute:
    - ```docker-compose up -d --build```

### **Workflow 🔄**:
* **Producer (producer.py)**:
    - Retrieves data from the data source and sends it to the Kafka Data Pipeline.
    - Initiates data transmission 10 seconds after the Docker container starts, ensuring stable environment and connections.
* **Consumer (consumer.py)**:
    - Reads data from the Kafka Data Pipeline and inserts it into the PostgreSQL database via database.py.
    - Starts 15 seconds after the Docker container runs to ensure proper synchronization with Kafka and API endpoints.
* **ETL Process (ETL.py)**:
    - Extracts raw data from the PostgreSQL database.
    - Transforms it into minute-interval tables (e.g., 10-minute summaries for temperature, 20-minute summaries for humidity).
    - Loads the transformed data back into separate PostgreSQL tables for efficient querying and analysis.
* **Delays**:
    - Delays between processes are intentional to ensure stable connections and mitigate synchronization issues. These delays can be adjusted based on performance needs.
### **Database Connection 🗄️**:
* Access the PostgreSQL database from the command line:
    - ```docker exec -it postgres psql -U admin -d kafka_data```
* To inspect tables in PostgreSQL:
    - ```SELECT * FROM synthetic_data;```
    - ```SELECT * FROM stock_price_summary;```
    - ```SELECT * FROM sales_trend_summary;```
### **Shutting Down 🔚**:
* Shut down the Docker application and remove volumes with:
    - ```docker-compose down -v```


# **Conclusion ✨**
This project successfully demonstrates the integration of real-time and batch data processing into a cohesive pipeline. By leveraging Apache Kafka, Docker, and PostgreSQL, the solution showcases a scalable architecture for data ingestion, processing, and reporting.

The modular design ensures each component operates independently, yet seamlessly, within a Docker-managed ecosystem. Real-time streaming with Kafka facilitates efficient data flow, while the ETL process organizes and optimizes data for analytical insights.

This project serves as a practical guide for building robust data pipelines and highlights the potential of modern tools to handle both real-time and batch processing requirements in a streamlined manner.