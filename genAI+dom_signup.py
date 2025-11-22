import streamlit as st
from huggingface_hub import InferenceClient
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait

# -----------------------------
# HF MODEL
# -----------------------------
hf_token = "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"
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
client = InferenceClient(model=MODEL, token=hf_token)


# -----------------------------
# SELENIUM DRIVER
# -----------------------------
def chrome_driver(headless=True, viewport=(1280, 800)):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument(f"--window-size={viewport[0]},{viewport[1]}")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    svc = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=svc, options=opts)


# -----------------------------
# FETCH DOM + ELEMENTS JSON
# -----------------------------
def fetch_dom_json(url):
    driver = chrome_driver(headless=True)
    try:
        driver.get(url)
        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return document.readyState === 'complete'")
        )
        time.sleep(2)

        elements_script = r"""
        const els = Array.from(document.querySelectorAll('body *'));
        function xpathOf(el){
            if(!el) return '';
            const parts = [];
            while(el && el.nodeType === Node.ELEMENT_NODE){
                let name = el.nodeName.toLowerCase();
                if(el.id){
                    parts.unshift(name + "[@id='" + el.id + "']");
                    break;
                }
                let idx = 1;
                let sib = el.previousElementSibling;
                while(sib){
                    if(sib.nodeName === el.nodeName) idx++;
                    sib = sib.previousElementSibling;
                }
                parts.unshift(name + '[' + idx + ']');
                el = el.parentElement;
            }
            return '/' + parts.join('/');
        }

        return els.map(el => {
            return {
                tag: el.tagName.toLowerCase(),
                id: el.id || "",
                classes: Array.from(el.classList || []),
                text: (el.innerText || '').trim().slice(0, 50),
                xpath: xpathOf(el)
            };
        });
        """
        return driver.execute_script(elements_script)

    finally:
        driver.quit()


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("GenAI Login/Signup Automation Generator")

user_url = st.text_input(
    "Enter Website URL",
    placeholder="https://www.linkedin.com/login"
)

if st.button("Generate Automation Code"):
    if not user_url.strip():
        st.error("Please enter a valid URL.")
        st.stop()

    with st.spinner("Fetching DOM using Selenium..."):
        raw_elements = fetch_dom_json(user_url)

    st.success(f"Extracted {len(raw_elements)} elements")

    # -----------------------------
    #  FILTER ONLY LOGIN/SIGNUP ELEMENTS
    # -----------------------------
    login_keywords = ["login", "sign", "email", "password", "user", "continue", "next", "submit"]

    filtered = []
    for el in raw_elements:
        txt = el["text"].lower()
        cid = el["id"].lower()
        cls = " ".join(el["classes"]).lower()

        if any(k in txt for k in login_keywords) or any(k in cid for k in login_keywords) or any(k in cls for k in login_keywords):
            filtered.append(el)

    filtered = filtered[:40]   # reduce size
    st.write(f"Relevant DOM elements: {len(filtered)}")

    # -----------------------------
    # COMPRESS JSON → ULTRA-SMALL FORMAT
    # -----------------------------
    compressed = []
    for el in filtered:
        compressed.append(
            f"{el['tag'].upper()} | id={el['id']} | text={el['text']} | classes={' '.join(el['classes'])} | xpath={el['xpath']}"
        )

    compressed_str = "\n".join(compressed)

    # -----------------------------
    # GENERATE SELENIUM CODE
    # -----------------------------
    with st.spinner("Generating Selenium Code using Llama 3..."):
        prompt = f"""
You are an expert QA automation engineer.

Here is a compressed list of DOM elements (only login/signup related):
{compressed_str}

Using ONLY this information:
1. Write a detailed step-by-step login/signup test case.
2. Then produce FULL Python Selenium code using:
   - webdriver.Chrome()
   - WebDriverWait + EC
   - Use only the ids/text/xpaths shown above
   - Keep browser open (no driver.quit())

Output format:
1. Test Case Steps
2. Python Code
"""

        response = client.chat_completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.5,
        )

        output = response.choices[0].message["content"]

    st.subheader("Generated Test Case + Automation Code")
    st.markdown(output)
