import streamlit as st
from streamlit_tree_select import tree_select
import pickle
from pathlib import Path
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture

class FoodSelector:
    def __init__(_self, food_list, 
                 tree_file = '01.Data/tree_structure_lima_10pct_2014_2023_400pca.json',
                 data_file = '01.Data/DB_nodes_alimento_Lima_10pct_2014_2023_with_embedding.pkl',
                 api_key = st.secrets['llm']['key_']): # el único input debe ser la lista de alimentos
        _self.tree = _self.load_tree(tree_file)
        _self.df = _self.load_data(data_file)
        _self.client = OpenAI(api_key=api_key)
        _self.food_list = food_list
        _self.initialize_session_state()

    def load_tree(_self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def load_data(_self, file_path):
        return pd.read_pickle(file_path)

    def initialize_session_state(_self):
        if 'food_index' not in st.session_state:
            st.session_state.food_index = 0
        if 'selected_foods' not in st.session_state:
            st.session_state.selected_foods = {}

    def get_embedding(_self, text, model="text-embedding-3-small"):
        if pd.isna(text):
            return [0]
        text = str(text).lower()
        return _self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def get_similarity(_self, x, emb):
        return 1 - distance.cosine(x, emb)

    def cluster_near_one_gmm(_self, series, n_components=2):
        cos = series.values.reshape(-1, 1)
        data = np.sqrt(2 - 2 * cos)
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(data)
        labels = gmm.predict(data)
        closest_cluster = min(set(labels), key=lambda label: abs(series[labels == label].mean() - 1))
        return labels == closest_cluster

    @st.cache_data(ttl=120)
    def find_food(_self, search_query):
        emb_buscar = _self.get_embedding(search_query)
        _self.df['similitud'] = _self.df['embedding'].apply(_self.get_similarity, args=(emb_buscar,))
        index_bool = _self.cluster_near_one_gmm(_self.df['similitud'])
        # return _self.df.loc[index_bool].sort_values(['similitud'], ascending=False).index.to_list()[:50]
        return _self.df.loc[index_bool].sort_values(['similitud'], ascending=False)['Alimento_name'].to_list()[:50]

    def filter_tree(_self, tree, codes):
        filtered_tree = []
        for node in tree:
            new_node = node.copy()
            if 'children' in node:
                new_children = _self.filter_tree(node['children'], codes)
                if new_children:
                    new_node['children'] = new_children
                    filtered_tree.append(new_node)
                elif node['value'] in codes:
                    new_node.pop('children', None)
                    filtered_tree.append(new_node)
            elif node['value'] in codes:
                filtered_tree.append(new_node)
        return filtered_tree
    
    def _generate_code(_self, foods_dct):
        added_schema = []
        code_added_schema = []
        code_remove_schema = []
        for food, selection in foods_dct.items():
            if selection:
                # # New Relationships
                # - `(:Hogar)-[:Consume]->(:Alimento)`
                # - `(:Alimento)-[:Pertenece_a]->(:Papa_nativa)`
                added_schema.append(f"- `(:Alimento)-[:Pertenece_a]->(:{food})`") # modificar para que sea ""
                # code_added_schema.append(f"CREATE (:{food})")
                code_added_schema_i = ("MATCH (a:Alimento) \n" \
	                    f"WHERE a.alimento_name IN {selection} \n" \
                        f"CREATE (a)-[:Pertenece_a]->(:{food})"
                        )
                code_added_schema.append(code_added_schema_i)
                # for item in selection:
                    # code_added_schema.append(f"CREATE (:{food})-[:CONTAINS]->(:{item})")
                # ejecutar todo o por cada alimento una vez?
                code_remove_schema.append(f"MATCH (p:{food}) \n" \
                                          "DETACH DELETE p")
        added_schema = "# New Relationships\n" + "\n".join(added_schema)
        return added_schema, code_added_schema, code_remove_schema

#############################
    def initialize_variables(_self):
        for i in range(len(_self.food_list)):
            current_food = _self.food_list[i]
            search_results = _self.find_food(current_food)
            st.session_state.selected_foods[current_food] = search_results
        foods_dct = st.session_state['selected_foods']
        added_schema, code_added_schema, code_remove_schema = _self._generate_code(foods_dct)
        return foods_dct, code_added_schema, added_schema, code_remove_schema
            # return current_food

    def run(_self):
        if st.session_state.food_index < len(_self.food_list):
            current_food = _self.food_list[st.session_state.food_index]
            with st.chat_message("assistant"):
                # esto se está ejecutando en cada cambio de multiselect?, no xq está con st.cache!
                search_query = st.text_input("Buscar alimentos", key=f"search_{current_food}")
                if search_query:
                    search_results = _self.find_food(search_query) # string
                else :
                    search_results = [] # string, pero debe ser index
                # current_selections = st.session_state.selected_foods[current_food] # string, pero debe ser index
                # if len(current_selections) == 0:
                # 1. mostrar los resultados en el tree
                selected_index_food = _self.df.loc[_self.df['Alimento_name'].isin(search_results)].index.to_list()
                subtree = _self.filter_tree(_self.tree, selected_index_food)
                selected_pre = tree_select(subtree, checked=selected_index_food, check_model='leaf')['checked']
                if st.button("Agregar", key=f"add_{current_food}"):
                    selected_pre_ints = [int(sel) for sel in selected_pre]
                    selected = [sel for sel in _self.df.loc[selected_pre_ints, 'Alimento_name']]
                    st.session_state.selected_foods[current_food] = st.session_state.selected_foods[current_food] + selected
                    st.success("Selecciones agregadas.")

                multiselect_selection_str = st.multiselect(
                    "Selecciona alimentos",
                    options=st.session_state.selected_foods[current_food],
                    default=st.session_state.selected_foods[current_food],
                    key=f"multiselect_{current_food}"
                )
                st.session_state.selected_foods[current_food] = multiselect_selection_str

                # if current_food!=search_results:
                #     checked_items = list(set(search_results))
                # else:
                #     checked_items = list(set(search_results) & set(current_selections)) # el problema es que cuando están seleccionado alimentos y se quiere hacer una búsqueda, la intersección elimina resultados de la b´suqeda
                #     # se debería agregar un botón para agregar los alimentos selccionados en la búsqueda
                # selected_index_food = _self.df.loc[_self.df['Alimento_name'].isin(checked_items)].index.to_list()
                # subtree = _self.filter_tree(_self.tree, selected_index_food)
                # selected_pre = tree_select(subtree, checked=selected_index_food, check_model='leaf')['checked']
                # # st.write(subtree)
                # selected_pre_ints = [int(sel) for sel in selected_pre]
                # # selected = [sel for sel in selected_pre]
                # selected = [sel for sel in _self.df.loc[selected_pre_ints, 'Alimento_name']]
                # current_selections = list(set(current_selections) - set(search_results) | set(selected))
                # st.session_state.selected_foods[current_food] = current_selections

                # multiselect_selection_str = st.multiselect(
                #     "Selecciona alimentos",
                #     # options=_self.df.loc[current_selections, 'Alimento_name'].to_list(),
                #     # default=_self.df.loc[current_selections, 'Alimento_name'].to_list(),
                #     options=current_selections,
                #     default=current_selections,
                #     key=f"multiselect_{current_food}"
                # )

                # multiselect_selection = _self.df.loc[_self.df['Alimento_name'].isin(multiselect_selection_str)]['Alimento_name'].to_list()
                # st.session_state.selected_foods[current_food] = multiselect_selection # string

                st.write(f"Alimento actual: **{current_food}**")

                # if st.button("Guardar", key=f"save_{current_food}"):
                #     st.success("Selecciones guardadas.")

                if st.button("Continuar", key=f"continue_{current_food}"):
                    print("-----CONTINUANDO-------")
                    st.session_state.food_index += 1
                    st.rerun()
        else:
            with st.chat_message("assistant"):
                st.write("¡Finalizamos!")
                st.write("Tus selecciones:")
                for food_item, selections in st.session_state.selected_foods.items():
                    st.write(f"Alimento: **{food_item}**")
                    # st.write(f"Selección: {', '.join(_self.df.loc[selections, 'Alimento_name'])}")
                    st.write(f"Selección: {', '.join(selections)}")
                st.session_state.continuar = True

        foods_dct = st.session_state['selected_foods']
        added_schema, code_added_schema, code_remove_schema = _self._generate_code(foods_dct)
        return foods_dct, code_added_schema, added_schema, code_remove_schema


# if __name__ == "__main__":
#     selector = FoodSelector(food_list=['Papa', 'Arroz', 'Frijol', 'Maíz', 'Trigo'])
#     selector.run()