def time_to_format(timestamp):
  # Helper vars:
  MINUTE = 60
  HOUR = MINUTE * 60
  DAY = HOUR * 24
  MONTH = DAY * 30

  # Get the days, hours, etc:
  months = int(timestamp / MONTH)
  days = int((timestamp / MONTH) / DAY)
  hours = int((timestamp % DAY) / HOUR)
  minutes = int((timestamp % HOUR) / MINUTE)
  seconds = int(timestamp % MINUTE)

  # Build up the pretty string (like this: "N days, N hours, N minutes, N seconds")
  string = ""
  if months > 0:
    string += str(months) + " " + (months == 1 and "month" or "months") + ", "
  if days > 0:
    string += str(days) + " " + (days == 1 and "day" or "days") + ", "
  if len(string) > 0 or hours > 0:
    string += str(hours) + " " + (hours == 1 and "hour" or "hours") + ", "
  if len(string) > 0 or minutes > 0:
    string += str(minutes) + " " + (minutes == 1 and "minute" or "minutes") + ", "
  string += str(seconds) + " " + (seconds == 1 and "second" or "seconds")

  return string