-- Fails if any incident appears more than once at the same landmark.
select log_id, landmark_min, count(*) as n
from {{ ref('fct_training_landmarks') }}
group by log_id, landmark_min
having count(*) > 1
