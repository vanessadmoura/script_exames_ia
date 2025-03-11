import os
import json
import datetime
import pandas as pd
import io
from google.cloud import pubsub_v1
from flask import Flask, send_file, jsonify
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = "sqlite:///exam_requests.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = scoped_session(sessionmaker(bind=engine))

class ExamRequest(Base):
    """
    Modelo de dados para armazenar solicitações de exames.

    Atributos:
        id (int): Identificador único da solicitação.
        solicitation_id (int): ID da solicitação.
        requester_name (str): Nome do solicitante.
        message_sent_date (datetime): Data em que a mensagem foi enviada.
        sent_message (str): Conteúdo da mensagem enviada.
        phone_number (str): Número de telefone do paciente.
        exame_name (str): Nome do exame solicitado.
        solicitation_date (str): Data da solicitação do exame.
        recipe_type (str): Tipo de receita médica associada.
    """

    __tablename__ = 'exam_requests'
    id = Column(Integer, primary_key=True, autoincrement=True)
    solicitation_id = Column(Integer, nullable=False)
    requester_name = Column(String, nullable=False)
    message_sent_date = Column(DateTime, nullable=False)
    sent_message = Column(String, nullable=False)
    phone_number = Column(String, nullable=False)
    exame_name = Column(String, nullable=False)
    solicitation_date = Column(String, nullable=False)
    recipe_type = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)

app = Flask(__name__)

@app.route("/export", methods=["GET"])
def export_xlsx():
    """
    Exporta os registros de solicitações de exames para um arquivo Excel (XLSX).

    Retorna:
        file: Arquivo XLSX contendo os registros da tabela exam_requests.
    """
    session = SessionLocal()
    try:
        query = session.query(ExamRequest)

        with engine.connect() as conn:
            df = pd.read_sql(query.statement, conn)

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='ExamRequests')

        excel_buffer.seek(0)

        return send_file(
            excel_buffer,
            as_attachment=True,
            download_name="exam_requests.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


def callback(message):
    """
    Processa mensagens recebidas do Google Pub/Sub e armazena os dados no banco de dados.

    Parâmetros:
        message (pubsub_v1.types.PubsubMessage): Mensagem recebida contendo os dados da solicitação.

    Ação:
        - Salva os dados da solicitação no banco, se ainda não existir um registro duplicado.
        - Confirma o processamento da mensagem (ack) ou a rejeita em caso de erro (nack).
    """
    session = SessionLocal()
    try:
        data = json.loads(message.data.decode("utf-8"))
        print(f"Enviando mensagem para {data['telefone']}...")

        if not all(key in data for key in ['solicitacao_id', 'telefone', 'nomes_dos_exames', 'solicitante', 'mensagem', 'data_da_solicitacao', 'tipo_de_receita']):
            raise ValueError("Dados incompletos na mensagem recebida.")

        for exame in data['nomes_dos_exames']:
            existing_record = session.query(ExamRequest).filter_by(
                solicitation_id=data['solicitacao_id'], phone_number=data['telefone'], exame_name=exame).first()

            if not existing_record:
                exam_request = ExamRequest(
                    solicitation_id=data['solicitacao_id'],
                    requester_name=data['solicitante'],
                    message_sent_date=datetime.datetime.now().date(),
                    sent_message=data['mensagem'],
                    phone_number=data['telefone'],
                    exame_name=exame,
                    solicitation_date=data['data_da_solicitacao'],
                    recipe_type=data['tipo_de_receita']
                )
                session.add(exam_request)

        session.commit()
        print("Mensagens salvas no banco de dados.")
        message.ack()
    except Exception as e:
        session.rollback()
        print(f"Erro ao processar a mensagem: {e}")
        message.nack()
    finally:
        session.close()


def listen_for_messages():
    """
    Inicia o consumidor do Google Pub/Sub e o servidor Flask.

    - Escuta mensagens na assinatura Pub/Sub.
    - Inicia o servidor Flask para exportação de dados via API.
    """

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.getenv("PROJECT_ID")
    subscription_id = os.getenv("PUBSUB_SUBSCRIPTION_ID")

    if not credentials_path or not project_id or not subscription_id:
        raise ValueError("Variáveis de ambiente necessárias não foram definidas.")

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    subscriber.subscribe(subscription_path, callback=callback)
    print("Escutando por mensagens no Pub/Sub...")

    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    listen_for_messages()
