
import datetime


date_format = '%Y%m%d%H%M%S'


def get_datetime_str(dt, timedifference):
    """

    Args:
      dt: 
      timedifference: 

    Returns:

    """
    # JST補正（UTC+9）
    date = dt + datetime.timedelta(hours=timedifference)
    return date, date.strftime(date_format)
