import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import json
from gcp_pub import mandar_messagem_para_pubsub


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_csv("sample_nao_estruturados_reduzido.csv")


def criar_mensagem_generica_cliente(nome_do_cliente, descricao):
    """
    Gera uma mensagem amigável para o cliente com base na descrição fornecida.
    """
    system_message = f"""
    Você é um secretário do consultório Folks que deve relembrar o cliente dos seus exames e encaminhamentos ou passar a prescrição para ele.

    Monte uma mensagem para enviar no WhatsApp do cliente seguindo essas regras:
    1) Cumprimente o Cliente pelo seu nome.
    2) Verifique se a Descrição fornecida é uma prescrição, encaminhamento ou exame.
    3) Passe as informações para o cliente de forma amigável.
    4) Incentive o cliente a realizar o exame/encaminhamento com a nossa clínica.

    Informações:
    Nome do Cliente : {nome_do_cliente}
    Descrição : {descricao}

    Se despeça de forma carinhosa e educada sem citar seu nome, representando o consultório Folks.
    Use emojis fofos, porém sem exageros. Não use a palavra Doutor/Doutora.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Descrição: {descricao}"}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message.content.strip()


def identificar_tipo_de_receita(descricao):
    """
    Identifica se a receita é uma Prescrição Médica, um Encaminhamento Médico ou um Exame Médico.
    Retorna apenas um desses três tipos.
    """
    system_message = f"""
    Você é um secretário do consultório Folks que deve verificar pela descrição qual o tipo de receita.

    Abaixo seguem os três tipos possíveis de receita:
    1) Prescrição Médica
    2) Encaminhamento Médico
    3) Exame Médico

    Retorne apenas uma das três opções sem nenhuma outra palavra.

    Informações:
    Descrição : {descricao}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Descrição: {descricao}"}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content.strip()


def identificar_exame_de_imagem(descricao):
    """
    Verifica se o exame é um exame de imagem e retorna um JSON contendo essa informação
    junto com o nome do exame.
    """
    system_message = f"""
    Você é um especialista em exames do consultório Folks que deve verificar se o exame é um exame de imagem.

    Retorne um JSON contendo:
    - Exame de imagem: "Sim" ou "Não"
    - Nome do exame

    Exemplo de saída:
    {{"Exame de imagem": "Sim", "Nome do exame": "Ultrassom"}}

    Não retorne a palavra Json antes do Json, retorne apenas o Json

    Informações:
    Descrição : {descricao}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Descrição: {descricao}"}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content.strip()


def enviar_mensagem_lembrete_exame_imagem(nome_cliente, tipo_exame):
    """
    Gera uma mensagem de lembrete amigável para exames de imagem pendentes.
    """
    system_message = f"""
    Você é um secretário do consultório Folks que deve relembrar o cliente dos seus exames de imagem pendentes.

    Monte uma mensagem para enviar no WhatsApp do cliente seguindo essas regras:
    1) Cumprimente o Cliente pelo seu nome.
    2) Relembre a importância de marcar seu exame de imagem pendente em nossa clínica.
    3) Passe as informações para o cliente de forma amigável.

    Informações:
    Nome do Cliente : {nome_cliente}
    Tipo e Nome do Exame : {tipo_exame}

    Se despeça de forma carinhosa e educada sem citar seu nome, representando o consultório Folks.
    Use emojis fofos, porém sem exageros. Não use a palavra Doutor/Doutora.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message.content.strip()


target_column = "DS_RECEITA"
for _, linha in df.iterrows():
    try:
        tipo_receita = identificar_tipo_de_receita(linha[target_column])
        
        if tipo_receita == "Exame Médico":
            tipo_exame = identificar_exame_de_imagem(linha[target_column])
  
            if tipo_exame and tipo_exame.strip().startswith("{") and tipo_exame.strip().endswith("}"):
                try:
                    tipo_exame = json.loads(tipo_exame)
                except json.JSONDecodeError as e:
                    print(f"Erro ao decodificar JSON: {e}")
                    continue
            else:
                print(f"Resposta não é um JSON válido ou está vazia: {tipo_exame}")
                continue

            if tipo_exame['Exame de imagem'] == "Sim":
                nome_cliente = linha['SOLICITANTE']
                solicitacao_id = linha['ID']
                telefone_cliente = linha['TEL']
                data_da_solicitacao = linha['DATA']

                mensagem_cliente = enviar_mensagem_lembrete_exame_imagem(nome_cliente, tipo_exame)
            
                mandar_messagem_para_pubsub({
                    "solicitacao_id": solicitacao_id,
                    "solicitante": nome_cliente,
                    "telefone": telefone_cliente,
                    "mensagem": mensagem_cliente,
                    "nomes_dos_exames": tipo_exame['Nome do exame'],
                    "data_da_solicitacao": data_da_solicitacao,
                    "tipo_de_receita": tipo_receita
                })
                print(f"Mensagem enviada para {nome_cliente} ({telefone_cliente}):\n{mensagem_cliente}\n")
            else:
                print(f"A receita do solicitante {linha['SOLICITANTE']} não é um exame de imagem pendente.")
        else:
            print(f"A receita do solicitante {linha['SOLICITANTE']} não é do tipo 'Exame Médico'.")
    except Exception as e:
        print(f"Erro ao processar a linha {linha['ID']}: {e}")

