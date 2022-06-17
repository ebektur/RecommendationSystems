from django import template
import html

register = template.Library()

@register.filter
def convert_byte_to_string(value):
	translationTable = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")
	modified_value = value[0].translate(translationTable)
	return modified_value if value else value