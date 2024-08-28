from typing import Any, Dict

import json
import regex as re # Use regex instead of re to used variable length lookbehind


def getJsonFromAnswer(answer: str) -> Dict[str, Any]:
    """
    This function parses an LLM response which is supposed to output a JSON object
    and optimistically searches for the substring containing the JSON object.
    """
    if not answer.strip().startswith("{"):
        # Find the start index of the actual JSON string
        # assuming the prefix is followed by the JSON object/array
        start_index = answer.find("{") if "{" in answer else answer.find("[")
        if start_index != -1:
            # Remove the prefix and any leading characters before the JSON starts
            answer = answer[start_index:]

    if not answer.strip().endswith("}"):
        # Find the end index of the actual JSON string
        # assuming the suffix is preceded by the JSON object/array
        end_index = answer.rfind("}") if "}" in answer else answer.rfind("]")
        if end_index != -1:
            # Remove the suffix and any trailing characters after the JSON ends
            answer = answer[: end_index + 1]

    # Handle weird escaped values. I am not sure why the model
    # is returning these, but the JSON parser can't take them
    answer = answer.replace(r"\_", "_")
    answer = answer.replace("\\n", "\n")
    # Remove https and http prefixes to not conflict with comment detection
    # Handle comments in the JSON response. Use regex from // until end of line
    # TODO: I am commenting this out for now because the regular expression
    #       mangles an otherwise correct JSON in the VLDB demo. See below for
    #       a copy of the string that it deforms if you'd like to modify the
    #       regex to work for its intended use case plus this one.
    #
    # answer = re.sub(r"(?<!https?:)\/\/.*?$", "", answer, flags=re.MULTILINE)
    answer = re.sub(r",\n.*\.\.\.$", "", answer, flags=re.MULTILINE)
    # Sanitize newlines in the JSON response
    answer = answer.replace("\n", " ")
    try:
        response = json.loads(answer)
    except Exception as e:
        if "items" in answer: # If we are in one to many
            # Find the last dictionary item not closed
            last_idx = answer.rfind("},")
            # Close the last dictionary item
            answer = answer[:last_idx+1] + "]}"
            response = json.loads(answer)
        else:
            raise e
    return response

# NOTE: JSON string which is broken when using https and http variable length lookbehind
# '{\n  "items": [\n    {\n      "authors": "Tomoya Suzuki, Kazuhiro Hiwada, Hirotsugu Kajihara, Shintaro Sano, Shuou Nomura, Tatsuo Shiozawa",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p1311-suzuki.pdf",\n      "title": "Approaching DRAM performance by using microsecond-latency flash memory for small-sized random read accesses: a new access method and its graph applications"\n    },\n    {\n      "authors": "Cheng Chen, Jun Yang, mian lu, taize wang, zhao zheng, Yuqiang Chen, Wenyuan Dai, Bingsheng He, Weng-Fai Wong, Guoan Wu, Yuping Zhao, Andy Rudoff",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p799-chen.pdf",\n      "title": "Optimizing In-memory Database Engine For AI-powered On-line Decision Augmentation Using Persistent Memory"\n    },\n    {\n      "authors": "Gang Liu, Leying Chen, Shimin Chen",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p835-liu.pdf",\n      "title": "Zen: a High-Throughput Log-Free OLTP Engine for Non-Volatile Main Memory"\n    },\n    {\n      "authors": "Baoyue Yan, Xuntao Cheng, Bo Jiang, Shibin Chen, Canfang Shang, Jianying Wang, kenry huang, Xinjun Yang, Wei Cao, Feifei Li",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p1872-yan.pdf",\n      "title": "Revisiting the Design of LSM-tree Based OLTP Storage Engine with Persistent Memory"\n    },\n    {\n      "authors": "Jong-Hyeok Park, Soyee Choi, Gihwan Oh, Sang Won Lee",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p1481-lee.pdf",\n      "title": "SaS: SSD as SQL Database System"\n    },\n    {\n      "authors": "Shaleen Deep, Anja Gruenheid, Paraschos Koutris, Jeff Naughton, Stratis Viglas",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p418-deep.pdf",\n      "title": "Comprehensive and Efficient Workload Compression"\n    },\n    {\n      "authors": "Sheng Wang, Yuan Sun, Zhifeng Bao",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p163-wang.pdf",\n      "title": "On the Efficiency of K-Means Clustering: Evaluation, Optimization, and Algorithm Selection"\n    },\n    {\n      "authors": "Zicun Cong, Lingyang Chu, Yu Yang, Jian Pei",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p1583-cong.pdf",\n      "title": "Comprehensible Counterfactual Explanation on Kolmogorov-Smirnov Test"\n    },\n    {\n      "authors": "Maximilian Schleich, Zixuan Geng, Yihong Zhang, Dan Suciu",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p1681-schleich.pdf",\n      "title": "GeCo: Quality Counterfactual Explanations in Real Time"\n    },\n    {\n      "authors": "Peng Cheng, Jiabao Jin, Lei Chen, Xuemin Lin, Libin Zheng",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p2177-cheng.pdf",\n      "title": "A Queueing-Theoretic Framework for Vehicle Dispatching in Dynamic Car-Hailing"\n    },\n    {\n      "authors": "Jakub Lemiesz",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p1655-lemiesz.pdf",\n      "title": "On the algebra of data sketches"\n    },\n    {\n      "authors": "Otmar Ertl",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p2244-ertl.pdf",\n      "title": "SetSketch: Filling the Gap between MinHash and HyperLogLog"\n    },\n    {\n      "authors": "Monica Chiosa, Thomas Preußer, Gustavo Alonso",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p2369-chiosa.pdf",\n      "title": "SKT: A One-Pass Multi-Sketch Data Analytics Accelerator"\n    },\n    {\n      "authors": "Fuheng Zhao, Sujaya A Maiyya, Ryan Weiner, Divy Agrawal, Amr El Abbadi",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p1215-zhao.pdf",\n      "title": "KLL±: Approximate Quantile Sketches over Dynamic Datasets"\n    },\n    {\n      "authors": "Yinda Zhang, Jinyang Li, Yutian Lei, Tong Yang, Zhetao Li, Gong Zhang, Bin Cui",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p128-zhang.pdf",\n      "title": "On-Off Sketch: A Fast and Accurate Sketch on Persistence"\n    },\n    {\n      "authors": "Nan Tang, Ju Fan, Fangyi Li, Jianhong Tu, Xiaoyong Du, Guoliang Li, Samuel Madden, Mourad Ouzzani",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p1254-tang.pdf",\n      "title": "RPT: Relational Pre-trained Transformer Is Almost All You Need towards Democratizing Data Preparation"\n    },\n    {\n      "authors": "Yiming Lin, Daokun Jiang, Roberto Yus, Georgios Bouloukakis, Andrew Chio, Sharad Mehrotra, Nalini Venkatasubramanian",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p329-lin.pdf",\n      "title": "LOCATER: Cleaning WiFi Connectivity Datasets for Semantic Localization"\n    },\n    {\n      "authors": "Yinjun Wu, James Weimer, Susan B Davidson",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p2410-wu.pdf",\n      "title": "CHEF: A Cheap and Fast Pipeline for Iteratively Cleaning Label Uncertainties"\n    },\n    {\n      "authors": "El Kindi Rezig, Mourad Ouzzani, Walid G. Aref, Ahmed Elmagarmid, Ahmed Mahmood, Michael Stonebraker",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p2546-rezig.pdf",\n      "title": "Horizon: Scalable Dependency-driven Data Cleaning"\n    },\n    {\n      "authors": "Zhiwei Chen, Shaoxu Song, Ziheng Wei, Jingyun Fang, Jiang Long",\n      "pdfLink": "http://vldb.org/pvldb/vol14/p2114-song.pdf",\n      "title": "Approximating Median Absolute Deviation with Bounded Error"\n    }\n  ]\n}'