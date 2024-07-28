from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from utils import get_relevant_chunks, rerank_chunks, print_chunks


def ask_legal_assistant(query, chat_history):
    model = ChatAnthropic(model="claude-3-sonnet-20240229")
    pinecone_namespace = "simple-rag-2-he"

    # Repharse query as a standalone question
    repharse_query_template = [
        ("system",
         "בהתחשב בשיחה הבאה, נסח מחדש את שאלת ההמשך כך שתהיה שאלה עצמאית, שהיא מובנת ומותאמת לחיפוש דמיון תוך שמירה על המשמעות. הפלט צריך להיות רק השאלה המנוסחת מחדש \n היסטוריית צ'אט: \n {chat_history}"),
        ("human", "מעקב: \n {question}")
    ]
    rephrase_query_prompt = ChatPromptTemplate.from_messages(
        repharse_query_template)
    chat_history_string = "\n".join(
        [f'{d["role"]}: {d["content"]}' for d in chat_history])
    rephrase_query_res = model.invoke(rephrase_query_prompt.format(
        question=query, chat_history=chat_history_string))
    rephrased_query = rephrase_query_res.content
    print("Rephrased Query: ", rephrased_query)
    # Get the relevant chunks of text
    relevant_chunks = get_relevant_chunks(query, pinecone_namespace, 10)
    relevant_chunks_text = [item["metadata"]["text"]
                            for item in relevant_chunks]
    rerank_results = rerank_chunks(query, relevant_chunks_text)
    most_relevant_chunks = rerank_results[0:3]
    relevant_chunks_text = "\n----------------------\n".join(
        [c["text"] for c in most_relevant_chunks])
    # print_chunks(most_relevant_chunks, True)

    # Get the Actual Answer from LLM
    get_answer_template = [
        ("system", """אתה חמורבי, מערכת ניתוח משפטית מתקדמת המתמחה בסוגיות משפטיות, אסטרטגיה משפטית, ליטיגציה ופיתוח עילות משפטיות. התפקיד שלך הוא לספק ניתוח מהיר, מדויק וחדשני המבוסס אך ורק על הידע שניתן לך. התמקד אך ורק בנושאים משפטיים.

 הנחיות עיקריות:

 1. ביס את כל התשובות שלך אך ורק על הידע שלך. עליך לבדוק בקפידה כל מידע, שם, מופע או תוכן מכל סוג שאתה מספק למשתמש. עליך לבדוק שוב כל פרט בפסק הדין או בסעיף החוק שאתה רושם למשתמש, ללא יוצא מן הכלל. אתה בהחלט לא חייב לבסס את עצמך על הנחות יסוד כדי ליצור תשובה. זכור, אתה מומחה ומקצוען, אז אתה לא צריך לעשות רק הנחות.

2. בסוגיות משפטיות, אתה רשאי, לפי שיקול דעתך, לדרג את התביעות בסולם של 1-10 עבור:
 - סמכות משפטית
 - ודאות עובדתית
 - חדשנות
 - יישומים
 - השפעה אפשרית
 - יחס עלות-תועלת

3. חשב ציון משוקלל: [(חוזק*1.2) + ודאות + חדשנות + ישימות + השפעה + עלות-תועלת] / 6.2, מעוגל לשני מקומות עשרוניים.

 4. הסבר למשתמש את המשמעות של כל אחת מההגדרות המרכיבות את ציון התביעה המשפטית ומדוע הערכת אותה כך. המלץ למשתמש על תיקונים, שינויים ושיפורים שיכולים לשפר את ציון ההערכה שלך.

 5. הרחב על כל תביעה משפטית, תוך הסבר הבסיס, ההיגיון וההשלכות שלה.

 6. הציעו פתרונות יצירתיים וחדשניים לעידוד דיון מעמיק.

 7. להגיב בתקיפות ובחוסר סבלנות לנושאים שאינם משפטיים, מבלי להיות פוגע.

 8. ציינו בבירור אי-ודאות בתשובתכם.

 9. בקש מידע נוסף אם חסרים פרטים חיוניים.

10. הודה ותקן מיד טעויות אם המשתמש מציין אותן.

11. הצלבת הפניות מרובות לאותו מקרה כדי להבטיח עקביות.

 12. אמת את כל הפרטים מול השאילתה המקורית לאחר זיהוי המקרה.

 תעדוף דיוק, מהירות וחשיבה חדשנית בתשובותיך.

יֶדַע:
{relevant_chunks}"""),
        ("human", "שְׁאֵלָה: {question}")
    ]
    get_answer_prompt = ChatPromptTemplate.from_messages(get_answer_template)
    print("Prompt: \n\n", get_answer_prompt.format(
        question=rephrased_query,
        relevant_chunks=relevant_chunks_text
    ))
    get_answer_res = model.invoke(get_answer_prompt.format(
        question=rephrased_query,
        relevant_chunks=relevant_chunks_text
    ))

    print("Answer: \n\n", get_answer_res)
    # return get_answer_res.content
    return {
        "rephrased_query": rephrased_query,
        "relevant_chunks": [c["text"] for c in most_relevant_chunks],
        "answer": get_answer_res.content,
    }


"""
ask_legal_assistant("what challenges are faced by them", [{
    "role": "human",
    "content": "Describe the hierarchy and organizational structure of the judicial system in Pakistan."
},
    {
    "role": "ai",
    "content": "The judicial system in Pakistan follows a hierarchical structure with the Supreme Court at the top, followed by High Courts in each province, and subordinate courts which include district and sessions courts, civil and judicial magistrates."
},
    {
    "role": "human",
    "content": "what is the role of subordinate courts"
},
    {
    "role": "ai",
    "content": "Subordinate courts handle the bulk of civil and criminal cases at the district level."
}
])
"""
