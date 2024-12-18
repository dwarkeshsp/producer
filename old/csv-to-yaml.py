import pandas as pd
import yaml
from pathlib import Path

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def convert_timestamps(df):
    examples = []
    for row in df['Timestamps'].dropna():
        examples.append(clean_text(row))
    return {
        'name': 'Timestamps Generator',
        'description': 'Generates timestamps for key moments in podcast episodes',
        'examples': examples
    }

def convert_titles(df):
    examples = []
    for title in df['Titles'].dropna():
        examples.append(clean_text(title))
    return {
        'name': 'Episode Titles',
        'description': 'Collection of episode titles',
        'examples': examples
    }

def convert_descriptions(df):
    examples = []
    for _, row in df.iterrows():
        if pd.notna(row['Tweet Text']):
            examples.append({
                'text': clean_text(row['Tweet Text']),
                'link': clean_text(row.get('Link', ''))
            })
    return {
        'name': 'Viral Episode Descriptions',
        'description': 'Viral-worthy episode descriptions for Twitter',
        'examples': examples
    }

def convert_titles_thumbnails(df):
    examples = []
    for _, row in df.iterrows():
        if pd.notna(row['Titles']) and pd.notna(row['Thumbnail']):
            examples.append({
                'title': clean_text(row['Titles']),
                'thumbnail': clean_text(row['Thumbnail'])
            })
    return {
        'name': 'Titles and Thumbnails',
        'description': 'Title and thumbnail combinations for episodes',
        'examples': examples
    }

def convert_viral_clips(df):
    examples = []
    for _, row in df.iterrows():
        if pd.notna(row['Tweet Text']) and pd.notna(row['Clip Transcript']):
            example = {
                'tweet': clean_text(row['Tweet Text']),
                'transcript': clean_text(row['Clip Transcript'])
            }
            if pd.notna(row.get('Link')):
                example['link'] = clean_text(row['Link'])
            if pd.notna(row.get('Likes')):
                example['metrics'] = {
                    'likes': int(row['Likes']),
                    'reposts': int(row['Reposts']),
                    'quotes': int(row['Quotes'])
                }
            examples.append(example)
    return {
        'name': 'Viral Clips',
        'description': 'Collection of viral clips with engagement metrics',
        'examples': examples
    }

def main():
    # Create prompts directory
    prompts_dir = Path('../prompts')
    prompts_dir.mkdir(exist_ok=True)
    print(f"Created prompts directory at {prompts_dir}")
    # Convert each CSV
    conversions = {
        'Timestamps.csv': (pd.read_csv('source/Timestamps.csv'), convert_timestamps),
        'Titles.csv': (pd.read_csv('source/Titles.csv'), convert_titles),
        'Viral Episode Descriptions.csv': (pd.read_csv('source/Viral Episode Descriptions.csv'), convert_descriptions),
        'Titles & Thumbnails.csv': (pd.read_csv('source/Titles & Thumbnails.csv'), convert_titles_thumbnails),
        'Viral Twitter Clips.csv': (pd.read_csv('source/Viral Twitter Clips.csv'), convert_viral_clips)
    }

    for filename, (df, converter) in conversions.items():
        output = converter(df)
        yaml_filename = prompts_dir / f"{filename.split('.')[0].lower().replace(' ', '_')}.yaml"

        with open(yaml_filename, 'w', encoding='utf-8') as f:
            yaml.dump(output, f, allow_unicode=True, sort_keys=False, width=1000)

        print(f"Converted {filename} to {yaml_filename}")

if __name__ == "__main__":
    main()
