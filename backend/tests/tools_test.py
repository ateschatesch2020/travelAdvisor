from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

load_dotenv()  # .env dosyasını yükle

events_db = [
    {
        "id": 101,
        "title": "Temel Seviye Yüzme Kursu",
        "category": "Spor",
        "start_date": "2026-06-01",
        "end_date": "2026-08-30",
        "status": "active",
        "capacity": 0,
    },
    {
        "id": 102,
        "title": "İleri Seviye React Workshop",
        "category": "Yazılım",
        "start_date": "2026-04-15",
        "end_date": "2026-04-20",
        "status": "active",
        "capacity": 15,
    },
    {
        "id": 103,
        "title": "Yağlı Boya Resim Teknikleri",
        "category": "Sanat",
        "start_date": "2026-05-10",
        "end_date": "2026-07-10",
        "status": "passive",
        "capacity": 10,
    },
    {
        "id": 104,
        "title": "Piyano Başlangıç Eğitimi",
        "category": "Müzik",
        "start_date": "2026-09-01",
        "end_date": "2026-12-25",
        "status": "active",
        "capacity": 8,
    },
    {
        "id": 105,
        "title": "Python ile Veri Analizi",
        "category": "Yazılım",
        "start_date": "2026-05-01",
        "end_date": "2026-06-15",
        "status": "active",
        "capacity": 25,
    },
    {
        "id": 106,
        "title": "Diksiyon ve Hitabet Kursu",
        "category": "Kişisel Gelişim",
        "start_date": "2026-04-01",
        "end_date": "2026-04-30",
        "status": "active",
        "capacity": 12,
    },
]

tavily_search = TavilySearch(
    max_results=3, 
    search_engine="google", 
    language="tr", 
    description="Kurs merkezi etkinlikleri hakkında bilgi aramak için kullanılır.")

@tool
def get_all_events():
    """
    Tüm etkinlikleri döndürür.
    """
    return events_db

@tool
def get_event_by_id(event_id):
    """
    Belirtilen ID'ye sahip etkinliği döndürür.
    """
    for event in events_db:
        if event["id"] == event_id:
            return event
    return None

@tool
def check_capacity(event_id):
    """
    Belirtilen ID'ye sahip etkinliğin kapasitesini kontrol eder.
    """
    for event in events_db:
        if event["id"] == event_id:
            return "BOŞ" if event["capacity"] > 0 else "DOLU"
    return "BULUNAMADI"

def invoke_with_user(user_id: str, question: str):
    thread_id = f"user_session_{user_id}"
    return agent.invoke({
        "messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}}
    )

tools = [get_all_events, get_event_by_id, check_capacity, tavily_search]

llm = ChatOpenRouter(model="openai/gpt-5.4-mini")

my_system_prompt = """
    Sen bir kurs merkezinde çalışan yapay zeka asistanısın. Kullanıcıların etkinliklerle ilgili sorularını yanıtlamakla görevlisin.

    Kurallar:
    1. Kullanıcıların sorularını anlamaya çalış ve uygun araçları kullanarak yanıtla.
    2. Eğer kullanıcı tüm etkinlikleri görmek istiyorsa, get_all_events aracını kullan.
    3. Eğer kullanıcı belirli bir etkinlik hakkında bilgi istiyorsa, get_event_by_id aracını kullanarak etkinliği bul.
    4. Eğer kullanıcı bir etkinliğin kapasitesinin dolup dolmadığını soruyorsa, check_capacity aracını kullanarak kapasite durumunu kontrol et.
    5. Eğer kullanıcı kurs merkezi etkinlikleri hakkında genel bilgi arıyorsa, tavily_search aracını kullanarak ilgili bilgileri bulmaya çalış.
    6. Yanıt verirken, kullanıcıya net ve anlaşılır bilgiler sunmaya çalış. Gereksiz detaylardan kaçın ve doğrudan sorunun cevabını ver.    
    """

checkpointer = MemorySaver()

agent = create_agent(
    model=llm, 
    tools=tools, 
    system_prompt=my_system_prompt,
    checkpointer=checkpointer)

response1 = invoke_with_user("1", "ID'si 102 olan etkinliğin kapasitesi dolmuş mu?")
response2 = invoke_with_user("1", "Kontentajı?")
response3 = invoke_with_user("1", "React nedir?")

print("1.Yanıt: ", response1["messages"][-1].content)
print("2.Yanıt: ", response2["messages"][-1].content)
print("3.Yanıt: ", response3["messages"][-1].content)


# thread_id = "user_session_1"

# # soru = "Bana tüm etkinlikleri gösterir misin?"
# soru = "ID'si 102 olan etkinliğin kapasitesi dolmuş mu?"

# response1 = agent.invoke({
#     "messages": [HumanMessage(content=soru)]},        
#     config={"configurable": {"thread_id": thread_id}
# })

# response2 = agent.invoke({
#     "messages": [HumanMessage(content="Kontentajı?")]},        
#     config={"configurable": {"thread_id": thread_id}
# })


