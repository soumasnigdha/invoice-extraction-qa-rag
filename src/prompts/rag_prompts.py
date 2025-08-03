from langchain.prompts import PromptTemplate

class RAGPrompts:
    def get_invoice_extraction_prompt(self) -> PromptTemplate:
        """Get RAG-enhanced invoice extraction prompt"""
        template = """You are an expert financial document analyzer with access to historical invoice data for context.

CONTEXT FROM SIMILAR INVOICES:
{context}

CURRENT INVOICE TO EXTRACT:
{current_document}

Instructions:
1. Use the context from similar invoices to understand common patterns and formats
2. Extract structured data from the CURRENT INVOICE only
3. Apply learned patterns from context to improve accuracy
4. Return data in the specified JSON structure
5. If a field is not found, use empty string or 0 as appropriate
6. For each line item, extract individual tax details (CGST, SGST, IGST, CESS)
7. Calculate total_amount = taxable_amount - discount_amount + all tax amounts

Extract the following information from the CURRENT INVOICE:
- Basic Invoice Information (invoice_number, invoice_date, due_date, etc.)
- Vendor/Supplier Information (name, address, contact details, tax numbers)  
- Customer Information (name, address, contact details)
- Line Items with individual tax breakdowns
- Payment Information (payment_terms, payment_method, bank_details)
- Overall Totals (tax_total, grand_total only)

Return ONLY a valid JSON object with this structure:
{{
    "invoice_info": {{
        "invoice_number": "",
        "invoice_date": "",
        "due_date": "",
        "po_number": "",
        "currency": ""
    }},
    "vendor": {{
        "name": "",
        "address": "",
        "phone": "",
        "email": "",
        "gst_number": "",
        "pan": ""
    }},
    "customer": {{
        "name": "",
        "address": "",
        "phone": "",
        "email": "",
        "gst_number": "",
        "pan": ""
    }},
    "line_items": [
        {{
            "item_name": "",
            "item_code": "",
            "hsn_sac_code": "",
            "quantity": 0,
            "taxable_amount": 0,
            "discount_amount": 0,
            "cgst_rate": 0,
            "cgst_amount": 0,
            "sgst_rate": 0,
            "sgst_amount": 0,
            "igst_rate": 0,
            "igst_amount": 0,
            "cess_rate": 0,
            "cess_amount": 0,
            "total_amount": 0
        }}
    ],
    "totals": {{
        "tax_total": 0,
        "grand_total": 0
    }},
    "payment_info": {{
        "payment_terms": "",
        "payment_method": "",
        "bank_account": "",
        "ifsc_code": ""
    }}
}}"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "current_document"]
        )
    
    def get_query_prompt(self) -> PromptTemplate:
        """Get prompt for querying document corpus"""
        template = """You are a financial document analysis assistant. Answer the user's query based on the provided context from the document corpus.

USER QUERY: {query}

RELEVANT CONTEXT FROM DOCUMENTS:
{context}

Instructions:
1. Answer the query based ONLY on the provided context
2. If the context doesn't contain enough information, clearly state this
3. Provide specific details and references when possible
4. Format your response clearly and professionally
5. Include relevant invoice numbers, dates, and amounts when available

Answer:"""
        return PromptTemplate(
            template=template,
            input_variables=["query", "context"]
        )
    
    def get_similarity_analysis_prompt(self) -> PromptTemplate:
        """Get prompt for analyzing document similarity"""
        template = """You are analyzing financial documents for similarity and patterns.

REFERENCE DOCUMENT:
{reference_document}

SIMILAR DOCUMENTS FOUND:
{similar_documents}

Instructions:
1. Analyze the key similarities between the reference document and found documents
2. Identify common patterns (vendors, amounts, formats, etc.)
3. Highlight any notable differences
4. Provide insights about document relationships

Analysis:"""
        return PromptTemplate(
            template=template,
            input_variables=["reference_document", "similar_documents"]
        )
    
    def get_mapping_prompt(self) -> PromptTemplate:
        """Get prompt for mapping extracted data to master template"""
        template = """You are an expert accountant. Map the extracted invoice data to the master accounting template.

EXTRACTED INVOICE DATA:
{invoice_data}

MASTER TEMPLATE FIELDS:
{template_fields}

Instructions:
1. Map the extracted data to appropriate template fields
2. For Transaction_Type, use "Purchase" for invoices from vendors
3. For Party_Type, use "Vendor" for supplier invoices
4. Set appropriate defaults for missing fields
5. For line items, map the first/main item details
6. Use proper date format (YYYY-MM-DD)
7. Return only a JSON object with the mapped data

Return the mapped data as a JSON object with all template fields filled:"""
        return PromptTemplate(
            template=template,
            input_variables=["invoice_data", "template_fields"]
        )
