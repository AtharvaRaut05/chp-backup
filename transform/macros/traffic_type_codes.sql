{# Incident types that can affect freeway traffic: collisions, hit-and-runs, hazards.
   Keep in sync with TRAFFIC_TYPE_CODES in src/common/config.py. #}
{% macro traffic_type_codes() -%}
'1179', '1180', '1181', '1182', '1183', '20001', '20002', '1125', '1125A'
{%- endmacro %}
