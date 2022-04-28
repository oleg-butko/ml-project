from forest_cover_type import settings


def create_sub_file(df):
    df.to_csv(settings.submission_fn, index=False)
