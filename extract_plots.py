import json
import sys

def main():
    path = "docs/Copy_of_EDA_Delhi_Electricity_Load.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    with open("plot_info.txt", "w", encoding="utf-8") as out:
        out.write("Graphs and Plots found in Notebook:\n")
        
        last_markdown = ""
        for i, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") == "markdown":
                last_markdown = "".join(cell.get("source", []))
            
            source = "".join(cell.get("source", []))
            if "plot" in source.lower() or "sns" in source.lower() or "chart" in source.lower() or "graph" in source.lower():
                out.write(f"--- Cell {i} ({cell.get('cell_type')}) ---\n")
                if last_markdown:
                    out.write(f"Context (Previous Markdown):\n{last_markdown}\n")
                    last_markdown = ""
                out.write(f"Code:\n{source}\n\n")

if __name__ == "__main__":
    main()
