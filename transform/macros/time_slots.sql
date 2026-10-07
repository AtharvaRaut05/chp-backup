{# 5-minute slot of the day (0-287) and weekday/weekend, from a Pacific-time timestamp. #}
{% macro slot_of_day(ts_local) -%}
(hour({{ ts_local }}) * 12 + minute({{ ts_local }}) / 5)
{%- endmacro %}

{% macro day_type(ts_local) -%}
if(day_of_week({{ ts_local }}) in (6, 7), 'weekend', 'weekday')
{%- endmacro %}
