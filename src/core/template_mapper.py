import json
from typing import Dict, Any
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import logging
import pandas as pd

class TemplateMapper:
    def __init__(self):
        from ..config import Config
        from ..prompts.rag_prompts import RAGPrompts
        
        self.config = Config()
        self.llm = Ollama(
            model=self.config.OLLAMA_MODEL,
            base_url=self.config.OLLAMA_BASE_URL,
            temperature=0.1
        )
        self.prompts = RAGPrompts()
        self.master_template = self._load_master_template()
        self.logger = logging.getLogger(__name__)
    
    def _load_master_template(self) -> Dict[str, Any]:
        """Load the master accounting template"""
        try:
            template_path = self.config.TEMPLATES_DIR / "master_template.json"
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            # Extract the field list and create a template dict
            field_list = template_data["Complete Accounting Template"]
            
            template = {}
            for field in field_list:
                # Set appropriate default values based on field type
                if any(keyword in field.lower() for keyword in ['rate', 'amount', 'quantity', 'price', 'salary']):
                    template[field] = 0
                elif any(keyword in field.lower() for keyword in ['date']):
                    template[field] = ""
                else:
                    template[field] = ""
            
            return template
            
        except Exception as e:
            self.logger.error(f"Error loading master template: {e}")
            return self._get_default_template()
    
    def map_to_template(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map extracted invoice data to master template"""
        try:
            # First try LLM-powered mapping
            llm_mapped_data = self._llm_map_to_template(invoice_data)
            
            # If LLM mapping fails, use rule-based fallback
            if not llm_mapped_data or 'error' in llm_mapped_data:
                self.logger.warning("LLM mapping failed, using rule-based fallback")
                return self._rule_based_mapping(invoice_data)
            
            # Fill missing fields with defaults
            final_mapped_data = self._fill_defaults(llm_mapped_data)
            
            return final_mapped_data
            
        except Exception as e:
            self.logger.error(f"Error mapping data: {e}")
            return self._rule_based_mapping(invoice_data)
    
    def _llm_map_to_template(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to map invoice data to master template"""
        try:
            # Prepare data for mapping
            invoice_json = json.dumps(invoice_data, indent=2)
            template_fields = list(self.master_template.keys())
            
            # Generate mapping prompt
            prompt = self.prompts.get_mapping_prompt()
            formatted_prompt = prompt.format(
                invoice_data=invoice_json,
                template_fields=template_fields
            )
            
            # Get LLM response
            response = self.llm(formatted_prompt)
            
            # Parse and validate response
            mapped_data = self._parse_mapping_response(response)
            
            return mapped_data
            
        except Exception as e:
            self.logger.error(f"Error in LLM mapping: {e}")
            return {}
    
    def _parse_mapping_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM mapping response"""
        import re
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in mapping response")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return {}
    
    def _rule_based_mapping(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based mapping when LLM fails"""
        mapped_data = self.master_template.copy()
        
        try:
            # Extract data sections
            invoice_info = invoice_data.get('invoice_info', {})
            vendor = invoice_data.get('vendor', {})
            customer = invoice_data.get('customer', {})
            totals = invoice_data.get('totals', {})
            payment_info = invoice_data.get('payment_info', {})
            line_items = invoice_data.get('line_items', [])
            
            # Map basic invoice information
            mapped_data.update({
                "Transaction_Type": "Purchase",
                "Category": "Invoice",
                "Date": invoice_info.get('invoice_date', ''),
                "Description": f"Invoice from {vendor.get('name', 'Unknown Vendor')}",
                "Reference_Number": invoice_info.get('invoice_number', ''),
                "Amount": totals.get('grand_total', 0),
                "Currency": invoice_info.get('currency', 'INR'),
                
                # Party (Vendor) Information
                "Party_Name": vendor.get('name', ''),
                "Party_Type": "Vendor",
                "Party_Address": vendor.get('address', ''),
                "Party_Phone": vendor.get('phone', ''),
                "Party_Email": vendor.get('email', ''),
                "Party_GST_Number": vendor.get('gst_number', ''),
                "Party_PAN": vendor.get('pan', ''),
                
                # Invoice specific fields
                "Invoice_Number": invoice_info.get('invoice_number', ''),
                "Invoice_Date": invoice_info.get('invoice_date', ''),
                "Due_Date": invoice_info.get('due_date', ''),
                "Payment_Terms": payment_info.get('payment_terms', ''),
                "Payment_Method": payment_info.get('payment_method', ''),
                "Bank_Account": payment_info.get('bank_account', ''),
                
                # Tax totals
                "Tax_Amount": totals.get('tax_total', 0),
                
                # Status fields
                "Payment_Status": "Pending",
                "Status": "Active",
                "Created_By": "RAG_System",
                "Modified_By": "RAG_System"
            })
            
            # Map line item information (use first item for main fields)
            if line_items:
                first_item = line_items[0]
                mapped_data.update({
                    "Item_Name": first_item.get('item_name', ''),
                    "Item_Code": first_item.get('item_code', ''),
                    "HSN_SAC_Code": first_item.get('hsn_sac_code', ''),
                    "Quantity": first_item.get('quantity', 0),
                    "Unit_Price": first_item.get('taxable_amount', 0),
                    "Discount_Amount": first_item.get('discount_amount', 0),
                    
                    # Tax details from first line item
                    "CGST_Rate": first_item.get('cgst_rate', 0),
                    "CGST_Amount": first_item.get('cgst_amount', 0),
                    "SGST_Rate": first_item.get('sgst_rate', 0),
                    "SGST_Amount": first_item.get('sgst_amount', 0),
                    "IGST_Rate": first_item.get('igst_rate', 0),
                    "IGST_Amount": first_item.get('igst_amount', 0),
                    "CESS_Rate": first_item.get('cess_rate', 0),
                    "CESS_Amount": first_item.get('cess_amount', 0),
                })
                
                # Set tax type based on tax structure
                if first_item.get('igst_rate', 0) > 0:
                    mapped_data["Tax_Type"] = "IGST"
                elif first_item.get('cgst_rate', 0) > 0 or first_item.get('sgst_rate', 0) > 0:
                    mapped_data["Tax_Type"] = "CGST+SGST"
                else:
                    mapped_data["Tax_Type"] = "No Tax"
            
            self.logger.info("Applied rule-based mapping successfully")
            return mapped_data
            
        except Exception as e:
            self.logger.error(f"Error in rule-based mapping: {e}")
            return self.master_template.copy()
    
    def _fill_defaults(self, mapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing fields with default values"""
        final_data = self.master_template.copy()
        final_data.update(mapped_data)
        
        # Ensure all template fields are present
        for field in self.master_template.keys():
            if field not in final_data:
                final_data[field] = self.master_template[field]
        
        return final_data
    
    def _get_default_template(self) -> Dict[str, Any]:
        """Get default template if loading fails"""
        return {
            "Transaction_Type": "",
            "Category": "",
            "Date": "",
            "Description": "",
            "Reference_Number": "",
            "Amount": 0,
            "Currency": "INR",
            "Party_Name": "",
            "Party_Type": "",
            "Party_Address": "",
            "Party_Phone": "",
            "Party_Email": "",
            "Party_GST_Number": "",
            "Party_PAN": "",
            "Invoice_Number": "",
            "Invoice_Date": "",
            "Due_Date": "",
            "Payment_Terms": "",
            "Payment_Method": "",
            "Payment_Status": "",
            "Status": ""
        }
    
    def validate_mapped_data(self, mapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean mapped data"""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check required fields
            required_fields = [
                "Transaction_Type", "Amount", "Party_Name", 
                "Invoice_Number", "Invoice_Date"
            ]
            
            for field in required_fields:
                if not mapped_data.get(field):
                    validation_results["errors"].append(f"Missing required field: {field}")
                    validation_results["is_valid"] = False
            
            # Validate data types
            numeric_fields = [
                "Amount", "Quantity", "Unit_Price", "Tax_Amount",
                "CGST_Rate", "CGST_Amount", "SGST_Rate", "SGST_Amount",
                "IGST_Rate", "IGST_Amount", "CESS_Rate", "CESS_Amount"
            ]
            
            for field in numeric_fields:
                if field in mapped_data:
                    try:
                        float(mapped_data[field])
                    except (ValueError, TypeError):
                        validation_results["warnings"].append(f"Invalid numeric value for {field}")
                        mapped_data[field] = 0
            
            # Validate dates
            date_fields = ["Date", "Invoice_Date", "Due_Date"]
            for field in date_fields:
                if field in mapped_data and mapped_data[field]:
                    # Simple date validation (you could enhance this)
                    if len(str(mapped_data[field])) < 8:
                        validation_results["warnings"].append(f"Invalid date format for {field}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating mapped data: {e}")
            validation_results["errors"].append(f"Validation error: {e}")
            validation_results["is_valid"] = False
            return validation_results
    
    def get_mapping_summary(self, original_data: Dict[str, Any], mapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of mapping process"""
        return {
            "original_fields_count": len(self._flatten_dict(original_data)),
            "mapped_fields_count": len(mapped_data),
            "template_fields_count": len(self.master_template),
            "mapping_timestamp": str(pd.Timestamp.now()),
            "key_mappings": {
                "invoice_number": f"{original_data.get('invoice_info', {}).get('invoice_number', '')} -> {mapped_data.get('Invoice_Number', '')}",
                "vendor_name": f"{original_data.get('vendor', {}).get('name', '')} -> {mapped_data.get('Party_Name', '')}",
                "total_amount": f"{original_data.get('totals', {}).get('grand_total', 0)} -> {mapped_data.get('Amount', 0)}"
            }
        }
    
    @staticmethod
    def _flatten_dict(d, parent_key='', sep='_'):
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(TemplateMapper._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(TemplateMapper._flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)
