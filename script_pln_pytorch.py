import pandas as pd
from gcp_pub import mandar_messagem_para_pubsub
from verifica_exame_imagem import retorna_se_e_exame_de_imagem_e_termos_encontrados
from modelo_verifica_tipo_de_receita.predict import identificar_tipo_de_receita


df = pd.read_csv("sample_nao_estruturados_reduzido.csv")

def formatar_mensagem_whatsapp(linha, exames_encontrados):
    """
    Formata a mensagem para envio no WhatsApp informando o cliente sobre os exames pendentes.
    """
    nome_cliente = linha['SOLICITANTE']
    exames = ", ".join(exames_encontrados) if exames_encontrados else "Nenhum exame identificado"

    mensagem = f"""
    Olá, {nome_cliente}! 👋

    Aqui é da Folks Consultório e temos algumas informações sobre os exames que você precisa fazer. 😊

    Exames encontrados:
    {exames}

    Lembre-se que realizar esses exames conosco é rápido, seguro e com um atendimento especializado! 🏥💙 Agende sua consulta aqui no Folks Consultório para garantir o melhor cuidado para a sua saúde.

    Qualquer dúvida, estamos à disposição!

    Abraços e até logo! 👋
    """
    return mensagem


for _, linha in df.iterrows():
    tipo_receita = identificar_tipo_de_receita(linha['DS_RECEITA'])
    
    if tipo_receita == "Exame Médico":
        exames_encontrados = retorna_se_e_exame_de_imagem_e_termos_encontrados(linha['DS_RECEITA'])
        nome_cliente = linha['SOLICITANTE']
        solicitacao_id = linha['ID']
        telefone_cliente = linha['TEL']
        data_da_solicitacao = linha['DATA']

        if exames_encontrados:
            mensagem_cliente = formatar_mensagem_whatsapp(linha, exames_encontrados)
            
            mandar_messagem_para_pubsub({
                "solicitacao_id": solicitacao_id,
                "solicitante": nome_cliente,
                "telefone": telefone_cliente,
                "mensagem": mensagem_cliente,
                "nomes_dos_exames": exames_encontrados,
                "data_da_solicitacao": data_da_solicitacao,
                "tipo_de_receita": tipo_receita
            })
            print(f"Mensagem enviada para {nome_cliente} ({telefone_cliente}):\n{mensagem_cliente}\n")
        else:
            print(f"O exame de {nome_cliente} não é um exame de imagem pendente.")
    else:
        print(f"A receita do solicitante {linha['SOLICITANTE']} não é um exame médico pendente.")
