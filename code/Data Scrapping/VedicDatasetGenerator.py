# Vedic Wisdom Dataset Generator
# This script creates comprehensive datasets for the Vedic RAG system from multiple sources

import os
import json
import requests
from huggingface_hub import hf_hub_download
import pandas as pd
from typing import List, Dict, Any
import time

class VedicDatasetGenerator:
    def __init__(self):
        self.output_dir = "nalanda_library"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_itihasa_dataset(self) -> List[Dict[str, Any]]:
        """Download the Itihasa dataset (Ramayana & Mahabharata)"""
        print("📥 Downloading Itihasa dataset (Ramayana & Mahabharata)...")
        
        try:
            # Download CSV files
            repo_id = "rahular/itihasa"
            
            files = {
                'train': ('train.en.csv', 'train.sn.csv'),
                'validation': ('dev.en.csv', 'dev.sn.csv'),
                'test': ('test.en.csv', 'test.sn.csv')
            }
            
            all_data = []
            
            for split, (en_file, sn_file) in files.items():
                print(f"  Downloading {split} files...")
                en_path = hf_hub_download(repo_id=repo_id, filename=en_file, repo_type="dataset")
                sn_path = hf_hub_download(repo_id=repo_id, filename=sn_file, repo_type="dataset")
                
                # Load CSV files safely
                en_data = self._load_csv_safely(en_path, 'en')
                sn_data = self._load_csv_safely(sn_path, 'sn')
                
                for i in range(min(len(en_data), len(sn_data))):
                    all_data.append({
                        'source': 'Itihasa',
                        'category': 'Ramayana' if i < len(en_data) * 0.2 else 'Mahabharata',  # Approximate split
                        'title': f"Verse {i+1}",
                        'sanskrit': sn_data.iloc[i]['sn'],
                        'english': en_data.iloc[i]['en'],
                        'verse': str(i+1)
                    })
            
            print(f"  ✅ Loaded {len(all_data)} verses from Itihasa dataset")
            return all_data
            
        except Exception as e:
            print(f"  ❌ Failed to download Itihasa dataset: {e}")
            return []
    
    def download_bhagavad_gita(self) -> List[Dict[str, Any]]:
        """Download Bhagavad Gita from a public API"""
        print("📥 Downloading Bhagavad Gita...")
        
        try:
            # Using a public Bhagavad Gita API
            url = "https://bhagavadgitaapi.in/chapters"
            response = requests.get(url)
            
            if response.status_code == 200:
                chapters = response.json()
                all_verses = []
                
                for chapter in chapters:
                    chapter_num = chapter['chapter_number']
                    verses_url = f"https://bhagavadgita.in/chapter/{chapter_num}"
                    
                    try:
                        verses_response = requests.get(verses_url)
                        if verses_response.status_code == 200:
                            # Parse verses from the HTML (this is a simplified approach)
                            # In practice, you might want to use a dedicated API
                            chapter_data = verses_response.text
                            
                            # Add chapter info
                            all_verses.append({
                                'source': 'Bhagavad Gita',
                                'category': 'Bhagavad Gita',
                                'title': f"Chapter {chapter_num}",
                                'sanskrit': chapter['name_transliterated'],
                                'english': chapter['name_meaning'],
                                'verse': f"Chapter {chapter_num}"
                            })
                    except:
                        continue
                
                print(f"  ✅ Loaded {len(all_verses)} chapters from Bhagavad Gita")
                return all_verses
            
        except Exception as e:
            print(f"  ❌ Failed to download Bhagavad Gita: {e}")
        
        # Fallback: Add some well-known Bhagavad Gita verses
        fallback_verses = [
            {
                'source': 'Bhagavad Gita',
                'category': 'Bhagavad Gita',
                'title': 'Chapter 2, Verse 47',
                'sanskrit': 'कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥',
                'english': 'You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty.',
                'verse': '2.47'
            },
            {
                'source': 'Bhagavad Gita',
                'category': 'Bhagavad Gita',
                'title': 'Chapter 2, Verse 20',
                'sanskrit': 'न जायते म्रियते वा कदाचिन्। नायं भूत्वा भविता वा न भूयः। अजो नित्यः शाश्वतोऽयं पुराणो। न हन्यते हन्यमाने शरीरे॥',
                'english': 'For the soul there is never birth nor death. It has not come into being, nor will it cease to be. It is unborn, eternal, ever-existing and primeval. It is not slain when the body is slain.',
                'verse': '2.20'
            }
        ]
        
        print(f"  ✅ Using fallback Bhagavad Gita verses: {len(fallback_verses)}")
        return fallback_verses
    
    def add_upanishads_sample(self) -> List[Dict[str, Any]]:
        """Add sample Upanishad verses"""
        print("📥 Adding Upanishad samples...")
        
        upanishad_verses = [
            {
                'source': 'Upanishads',
                'category': 'Upanishad',
                'title': 'Isha Upanishad, Verse 1',
                'sanskrit': 'पूर्णमदः पूर्णमिदं पूर्णात् पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥',
                'english': 'That is whole, this is whole; the whole has come out of the whole. When the whole is taken from the whole, the whole remains.',
                'verse': 'Isha 1'
            },
            {
                'source': 'Upanishads',
                'category': 'Upanishad',
                'title': 'Katha Upanishad, Verse 2.1.1',
                'sanskrit': 'उत्तिष्ठत जाग्रत प्राप्य वरान् निबोधत। क्षुरस्य धारा निशिता दुरत्यया दुर्गं पथस्तत् कवयो वदन्ति॥',
                'english': 'Arise! Awake! Having approached the eminent teachers, understand them. The path is as sharp as the edge of a razor, and thus difficult to traverse, say the wise.',
                'verse': 'Katha 2.1.1'
            }
        ]
        
        print(f"  ✅ Added {len(upanishad_verses)} Upanishad verses")
        return upanishad_verses
    
    def add_vedas_sample(self) -> List[Dict[str, Any]]:
        """Add sample Vedic verses"""
        print("📥 Adding Vedic samples...")
        
        vedic_verses = [
            {
                'source': 'Vedas',
                'category': 'Veda',
                'title': 'Rig Veda, Mandala 1, Sukta 1',
                'sanskrit': 'अग्निमीळे पुरोहितं यज्ञस्य देवं रत्वीजम्। होतारं रत्नधातमम्॥',
                'english': 'I praise Agni, the chosen priest, god of the sacrifice, the minister of the rite, the bestower of treasures.',
                'verse': 'RV 1.1.1'
            },
            {
                'source': 'Vedas',
                'category': 'Veda',
                'title': 'Yajur Veda, Taittiriya Samhita',
                'sanskrit': 'शं नो मित्रः शं वरुणः शं भवत्यर्यमा। शं इन्द्रो बृहस्पतिः शं विष्णुरुक्रमः॥',
                'english': 'May Mitra be kind to us, and Varuna, Aryaman, Indra, Brihaspati, and the wide-striding Vishnu.',
                'verse': 'YV TS 1.1.1'
            }
        ]
        
        print(f"  ✅ Added {len(vedic_verses)} Vedic verses")
        return vedic_verses
    
    def _load_csv_safely(self, file_path: str, column_name: str) -> pd.DataFrame:
        """Load CSV with error handling"""
        try:
            df = pd.read_csv(file_path, header=None, names=[column_name])
            return df
        except:
            try:
                df = pd.read_csv(file_path, header=None, names=[column_name], sep='\t')
                return df
            except:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                data = [[line.strip()] for line in lines if line.strip()]
                return pd.DataFrame(data, columns=[column_name])
    
    def create_dataset_files(self, all_data: List[Dict[str, Any]]):
        """Create formatted text files for RAG system"""
        print(f"📝 Creating dataset files from {len(all_data)} total verses...")
        
        # Split into manageable files
        max_entries_per_file = 20000
        file_index = 1
        entry_count = 0
        
        current_file = open(f"{self.output_dir}/nalanda_corpus_part_{file_index}.txt", "w", encoding="utf-8")
        
        # Create metadata file
        metadata = {
            'total_verses': len(all_data),
            'sources': {},
            'categories': {},
            'files_created': []
        }
        
        for verse in all_data:
            # Format for RAG consumption
            chunk = f"Source: {verse['source']}\n"
            chunk += f"Category: {verse['category']}\n"
            chunk += f"Title: {verse['title']}\n"
            chunk += f"Verse: {verse['verse']}\n"
            chunk += f"Sanskrit: {verse['sanskrit']}\n"
            chunk += f"English: {verse['english']}\n"
            chunk += "---\n"
            
            current_file.write(chunk)
            
            # Update metadata
            source = verse['source']
            category = verse['category']
            metadata['sources'][source] = metadata['sources'].get(source, 0) + 1
            metadata['categories'][category] = metadata['categories'].get(category, 0) + 1
            
            entry_count += 1
            if entry_count >= max_entries_per_file:
                current_file.close()
                metadata['files_created'].append(f"nalanda_corpus_part_{file_index}.txt")
                file_index += 1
                current_file = open(f"{self.output_dir}/nalanda_corpus_part_{file_index}.txt", "w", encoding="utf-8")
                entry_count = 0
                print(f"  Created file {file_index-1}, starting file {file_index}...")
        
        current_file.close()
        metadata['files_created'].append(f"nalanda_corpus_part_{file_index}.txt")
        
        # Save metadata
        with open(f"{self.output_dir}/dataset_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Success! Created {file_index} text files in '{self.output_dir}' folder.")
        print(f"📊 Dataset Summary:")
        for source, count in metadata['sources'].items():
            print(f"  {source}: {count} verses")
        for category, count in metadata['categories'].items():
            print(f"  {category}: {count} verses")
    
    def generate_comprehensive_dataset(self):
        """Generate the complete Vedic dataset"""
        print("🕉️ Starting Vedic Wisdom Dataset Generation...")
        print("=" * 60)
        
        all_data = []
        
        # Collect data from all sources
        all_data.extend(self.download_itihasa_dataset())
        time.sleep(1)  # Rate limiting
        
        all_data.extend(self.download_bhagavad_gita())
        time.sleep(1)
        
        all_data.extend(self.add_upanishads_sample())
        all_data.extend(self.add_vedas_sample())
        
        if all_data:
            self.create_dataset_files(all_data)
            print("\n🎉 Vedic dataset generation completed!")
            print(f"📁 Files saved in: {self.output_dir}/")
            print("📄 Ready for RAG system integration!")
        else:
            print("❌ No data was collected. Please check your internet connection.")

if __name__ == "__main__":
    generator = VedicDatasetGenerator()
    generator.generate_comprehensive_dataset()
