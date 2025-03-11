from google.cloud import pubsub_v1
import json
import os
from dotenv import load_dotenv


load_dotenv()


def mandar_messagem_para_pubsub(exam_request_data):
    """
    Envia uma mensagem para um tópico do Google Pub/Sub.

    Parâmetros:
        exam_request_data (dict): Dicionário contendo os dados da solicitação de exame.
            - solicitacao_id (int): ID da solicitação.
            - telefone (str): Número de telefone do solicitante.
            - Outros campos relevantes para a mensagem.

    Retorna:
        None. A função publica a mensagem no tópico Pub/Sub e imprime uma confirmação.
    """
    exam_request_data['solicitacao_id'] = int(exam_request_data['solicitacao_id'])
    exam_request_data['telefone'] = str(exam_request_data['telefone'])

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("PROJECT_ID")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    topic_id = "mensagens-whatsapp"  

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    message_data = json.dumps(exam_request_data)
    message_data = message_data.encode("utf-8")

    future = publisher.publish(topic_path, message_data)
    print(f"Mensagem enviada para o Pub/Sub: {future.result()}")
