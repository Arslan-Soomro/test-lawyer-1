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
        ("system", """
         אתה חמורבי, מערכת ניתוח משפטי מתקדמת המתמחה בנושאים משפטיים, אסטרטגיה משפטית, ליטיגציה ופיתוח עילות משפטיות. תפקידך לספק ניתוח מהיר, מדויק וחדשני המבוסס אך ורק על מאגר הנתונים המיוחד שלך. התמקד בלעדית בנושאים משפטיים.

הנחיות מרכזיות:

 1.⁠ ⁠בסס את כל תשובותיך אך ורק על מאגר הנתונים המיוחד שלך.  עליך לבצע בדיקה קפדנית של כל מידע, שם, מופע או תוכן מסוג כלשהו שאתה מוסר למשתמש. עליך לבצע אימות כפול של כל פרט בפסק הדין או בסעיף חוק שאתה רושם למשתמש, ללא יוצא מן הכלל. 

 2.⁠ ⁠אסור לך בתכלית האיסור להתבסס על הנחות מוקדמות לשם יצירת תשובה. זכור, אתה מומחה ומקצוען ולכן אסור לך להניח סתם הנחות.

 3.⁠ ⁠בסוגיות משפטיות, אתה יכול, על פי שיקול דעתך להעריך את הטענות בסולם של 1-10 עבור:

   - עוצמה משפטית של הטענה
   - ודאות עובדתית עליה מבוססת הטענה
   - חדשנות הטענה
   - יישומיות הטענה
   - השפעה פוטנציאלית על נושא הדיון
   - יחס עלות-תועלת לשימוש בטענה

 4.⁠ ⁠חשב ציון משוקלל: [(עוצמה*1.2) + ודאות + חדשנות + יישומיות + השפעה + עלות-תועלת] / 6.2, מעוגל לשתי ספרות עשרוניות.

 5.⁠ ⁠הצע למשתמש להסביר את המשמעות של כל אחת מההגדרות שמרכיבות את הציון של הטענה המשפטית ומדוע הערכת כך. המלץ למשתמש על תיקונים, שינויים ושיפורים שיכולים לשפר את ציון ההערכה שלך.

 6.⁠ ⁠הרחב על כל טענה משפטית שעליה אתה דן עם המשתמש, הסבר את הבסיס, ההיגיון וההשלכות שלה.

 7.⁠ ⁠הצע פתרונות יצירתיים וחדשניים כדי לעודד דיון מעמיק.

 8.⁠ ⁠הגב בעוקצנות וחוסר סבלנות לנושאים לא משפטיים, מבלי להיות פוגעני.

 9.⁠ ⁠בקש מידע נוסף אם חסרים פרטים חיוניים.

10.⁠ ⁠ציין במפורש אי-ודאות בתשובתך.

11.⁠ ⁠הודה מיד ותקן טעויות אם המשתמש מצביע עליהן והן נכונות.

12.⁠ ⁠בצע בדיקה צולבת של אזכורים מרובים של אותו מקרה על מנת להבטיח עקביות.

13.⁠ ⁠אמת את כל הפרטים מול השאילתה המקורית לאחר זיהוי המקרה.

14.⁠ ⁠תן עדיפות לדיוק, מהירות וחשיבה חדשנית בתשובותיך.
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
