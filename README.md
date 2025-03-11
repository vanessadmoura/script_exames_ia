# 🚀 Desafio - Usando IA para enviar mensagens de WhatsApp personalizadas

## Tecnologias usadas:

- **Pytorch, Sklearn**: Treinamento de modelo para identificar o tipo da receita
- **Flask, SQLAlchemy**: API para receber mensagens do Google Pub/Sub e realizar o envio da mensagem para o cliente
- **Google Pub/Sub GCP**: Mensageria
- **Regex (Expressões Regulares)**: Utilização de expressões regulares para processamento de dados
- **OpenAI (LLM) com LangChain**: Utilização de modelos de linguagem para personalização e automação de mensagens
- **Pandas**: Manipulação de dados

## ⚙️ Configuração do ambiente

- Verifique se todos as credenciais necessárias estão no .env
- Crie um ambiente virtual, usando o seguinte comando:

### No Windows:

```bash
python -m venv venv
source venv/scripts/activate
```

### No MacOS e Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

- Instale todas as dependências do requirements.txt, usando o seguinte comando:

```bash
pip install -r requirements.txt
```

## 🧠 Como rodar o treinamento do modelo e validar usando o predict

- Para iniciar o treinamento, basta executar o seguinte comando (Já deixei o arquivo no repositório, não é necessário rodar novamente):

```bash
python modelo_verifica_tipo_de_receita/train.py
```

- Para testar o modelo, basta descomentar os prints das últimas linhas do arquivo predict.py (o método: identificar_tipo_de_receita) e executar o seguinte comando:

```bash
python modelo_verifica_tipo_de_receita/predict.py
```

## 🔍 Como testar a extração dos nomes dos exames de imagem

- Para testar o Regex, basta descomentar os prints das últimas linhas do arquivo verifica_exame_imagem.py e executar o seguinte comando:

```bash
python verifica_exame_imagem.py
```

## ⏯️ Como rodar os scripts:

- Para executar o script de identificação e criação de mensagens que utiliza PLN, execute o seguinte comando:

```bash
python script_pln_pytorch.py
```

- Para executar o script de identificação e criação de mensagens que utiliza LLM ChatGPT4o, execute o seguinte comando:

```bash
python script_llm.py
```

- Ao executar os scripts, eles salvarão os dados das mensagens e dos clientes no Google Pub/Sub

## 📡 Como rodar a API que realiza o envio das mensagens:

- Essa API ao ser iniciada, escutará e receberá os dados e mensagens do Google Pub/Sub e enviará a mensagem para o WhatsApp e salvará os dados no Banco de Dados
- Para executar a API, utilize o seguinte comando:

```bash
python api_envio_whatsapp/app.py
```