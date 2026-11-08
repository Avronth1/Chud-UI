import copy
import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# ==================== CLASS LOGIC (FastGraphApp) ====================
class FastGraphApp:
    def __init__(self):
        self.nodes = []          # Danh sách tên các điểm (index là ID)
        self.adj_matrix = []     # Ma trận kề không trọng số
        self.weight_matrix = []  # Ma trận kề có trọng số (khoảng cách)
        self.history = []        # Ngăn xếp lưu lịch sử trạng thái để Undo

    def save_state(self):
        state = {
            'nodes': copy.deepcopy(self.nodes),
            'adj_matrix': copy.deepcopy(self.adj_matrix),
            'weight_matrix': copy.deepcopy(self.weight_matrix)
        }
        self.history.append(state)

    def undo(self):
        if not self.history:
            return False, "Không còn lịch sử để Hoàn tác (Undo)!"
        prev_state = self.history.pop()
        self.nodes = prev_state['nodes']
        self.adj_matrix = prev_state['adj_matrix']
        self.weight_matrix = prev_state['weight_matrix']
        return True, "Đã hoàn tác (Undo) về trạng thái trước đó!"

    def add_node(self, name):
        self.save_state()
        self.nodes.append(name)
        n = len(self.nodes)
        
        for row in self.adj_matrix:
            row.append(0)
        self.adj_matrix.append([0] * n)

        for row in self.weight_matrix:
            row.append(0.0)
        self.weight_matrix.append([0.0] * n)

    def remove_node(self, idx):
        if not (0 <= idx < len(self.nodes)):
            return False, "ID điểm không tồn tại!"
        
        self.save_state()
        removed_name = self.nodes.pop(idx)
        self.adj_matrix.pop(idx)
        for row in self.adj_matrix:
            row.pop(idx)

        self.weight_matrix.pop(idx)
        for row in self.weight_matrix:
            row.pop(idx)
        return True, f"Đã xóa điểm [{idx}] '{removed_name}' thành công."

    def clear_all_nodes(self):
        self.save_state()
        self.nodes = []
        self.adj_matrix = []
        self.weight_matrix = []

    def add_edge_by_id(self, u, v, weight=1.0):
        n = len(self.nodes)
        if 0 <= u < n and 0 <= v < n:
            self.save_state()
            self.adj_matrix[u][v] = 1
            self.adj_matrix[v][u] = 1
            self.weight_matrix[u][v] = weight
            self.weight_matrix[v][u] = weight
            return True, f"Đã nối cạnh [{u}: {self.nodes[u]}] <-> [{v}: {self.nodes[v]}] (Khoảng cách: {weight})."
        return False, "ID điểm không hợp lệ!"

    def remove_edge_by_id(self, u, v):
        n = len(self.nodes)
        if 0 <= u < n and 0 <= v < n:
            if self.adj_matrix[u][v] == 1:
                self.save_state()
                self.adj_matrix[u][v] = 0
                self.adj_matrix[v][u] = 0
                self.weight_matrix[u][v] = 0.0
                self.weight_matrix[v][u] = 0.0
                return True, f"Đã xóa cạnh nối giữa [{u}: {self.nodes[u]}] và [{v}: {self.nodes[v]}]."
            else:
                return False, "Giữa hai điểm này không có cạnh nối!"
        return False, "ID điểm không hợp lệ!"

    def clear_all_edges(self):
        self.save_state()
        n = len(self.nodes)
        self.adj_matrix = [[0] * n for _ in range(n)]
        self.weight_matrix = [[0.0] * n for _ in range(n)]

    def solve_euler(self):
        n = len(self.nodes)
        if n == 0:
            return "Đồ thị trống!"

        degrees = [sum(row) for row in self.adj_matrix]
        odd_nodes = [i for i, deg in enumerate(degrees) if deg % 2 != 0]

        res = []
        if len(odd_nodes) == 0:
            res.append("-> **Đồ thị có chu trình Euler** (Mọi đỉnh đều có bậc chẵn).")
        elif len(odd_nodes) == 2:
            res.append(f"-> **Đồ thị có đường đi Euler** (Có đúng 2 đỉnh bậc lẻ: [{odd_nodes[0]}] {self.nodes[odd_nodes[0]]}, [{odd_nodes[1]}] {self.nodes[odd_nodes[1]]}).")
        else:
            return "-> **Đồ thị KHÔNG có chu trình hay đường đi Euler.**"

        adj = [row[:] for row in self.adj_matrix]
        start_idx = odd_nodes[0] if len(odd_nodes) == 2 else 0
        curr_path = [start_idx]
        circuit = []

        while curr_path:
            curr = curr_path[-1]
            next_node = -1
            for j in range(n):
                if adj[curr][j] > 0:
                    next_node = j
                    break
            if next_node != -1:
                adj[curr][next_node] -= 1
                adj[next_node][curr] -= 1
                curr_path.append(next_node)
            else:
                circuit.append(curr_path.pop())

        path_reversed = circuit[::-1]
        path_str = " -> ".join([f"[{i}] {self.nodes[i]}" for i in path_reversed])
        
        total_dist = 0
        for i in range(len(path_reversed) - 1):
            u = path_reversed[i]
            v = path_reversed[i+1]
            total_dist += self.weight_matrix[u][v]

        res.append(f"**Đường đi Euler:** {path_str}")
        res.append(f"**Tổng khoảng cách:** {total_dist}")
        return "\n\n".join(res)

    def solve_hamilton(self, start_idx, end_idx):
        n = len(self.nodes)
        if not (0 <= start_idx < n and 0 <= end_idx < n):
            return "ID điểm đầu hoặc điểm cuối không hợp lệ!"

        path = [-1] * n
        visited = [False] * n
        path[0] = start_idx
        visited[start_idx] = True

        def backtrack(pos):
            if pos == n:
                return path[pos - 1] == end_idx

            for v in range(n):
                if self.adj_matrix[path[pos - 1]][v] > 0 and not visited[v]:
                    path[pos] = v
                    visited[v] = True
                    if backtrack(pos + 1):
                        return True
                    visited[v] = False
                    path[pos] = -1
            return False

        if backtrack(1):
            path_str = " -> ".join([f"[{i}] {self.nodes[i]}" for i in path])
            total_dist = 0
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                total_dist += self.weight_matrix[u][v]

            res = [
                f"**Kết quả Hamilton từ [{start_idx}] {self.nodes[start_idx]} đến [{end_idx}] {self.nodes[end_idx]}:**",
                f"**Đường đi:** {path_str}",
                f"**Tổng khoảng cách:** {total_dist}"
            ]
            return "\n\n".join(res)
        else:
            return f"Không tìm thấy đường đi Hamilton phù hợp giữa hai điểm này."


# ==================== STREAMLIT UI ====================
st.set_page_config(layout="wide", page_title="Graph Visualizer & Solver", page_icon="🌐")

if 'app' not in st.session_state:
    st.session_state.app = FastGraphApp()

app = st.session_state.app

st.title("🌐 Quản lý & Trực quan hóa Đồ thị Tương tác")

col_control, col_visual = st.columns([1, 2])

with col_control:
    # Nút Hoàn tác & Reset nhanh
    col_undo, col_clear = st.columns(2)
    with col_undo:
        if st.button("↩️ Hoàn tác (Undo)", use_container_width=True):
            ok, msg = app.undo()
            if ok:
                st.toast(msg)
            else:
                st.warning(msg)
            st.rerun()
    with col_clear:
        if st.button("🗑️ Xóa toàn bộ Đồ thị", use_container_width=True):
            app.clear_all_nodes()
            st.toast("Đã xóa toàn bộ đồ thị!")
            st.rerun()

    # Bộ điều khiển chuyển Tab các tính năng
    tab_nodes, tab_edges, tab_matrices, tab_algorithms = st.tabs([
        "📍 Đỉnh", "🔗 Cạnh", "📊 Ma Trận", "🧩 Thuật Toán"
    ])

    # --- TAB 1: QUẢN LÝ ĐỈNH ---
    with tab_nodes:
        st.subheader("Thêm Đỉnh Mới")
        with st.form("add_node_form", clear_on_submit=True):
            new_node_name = st.text_input("Tên điểm mới")
            if st.form_submit_button("Thêm điểm") and new_node_name.strip():
                app.add_node(new_node_name.strip())
                st.success(f"Đã thêm đỉnh: {new_node_name.strip()}")
                st.rerun()

        if app.nodes:
            st.subheader("Xóa Đỉnh")
            node_opts = {f"[{i}] {name}": i for i, name in enumerate(app.nodes)}
            selected_node_to_del = st.selectbox("Chọn đỉnh muốn xóa", list(node_opts.keys()))
            if st.button("Xóa đỉnh đã chọn", type="secondary"):
                idx = node_opts[selected_node_to_del]
                ok, msg = app.remove_node(idx)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    # --- TAB 2: QUẢN LÝ CẠNH ---
    with tab_edges:
        if len(app.nodes) < 2:
            st.info("Cần ít nhất 2 đỉnh để thực hiện thao tác trên cạnh!")
        else:
            st.subheader("Nối Cạnh Mới")
            node_opts = {f"[{i}] {name}": i for i, name in enumerate(app.nodes)}
            
            with st.form("add_edge_form"):
                u_label = st.selectbox("Điểm thứ nhất", list(node_opts.keys()), key="u_edge")
                v_label = st.selectbox("Điểm thứ hai", list(node_opts.keys()), key="v_edge")
                weight = st.number_input("Khoảng cách / Trọng số", min_value=0.1, value=1.0, step=0.5)
                
                if st.form_submit_button("Nối cạnh"):
                    u_idx = node_opts[u_label]
                    v_idx = node_opts[v_label]
                    if u_idx == v_idx:
                        st.warning("Không thể nối một điểm với chính nó!")
                    else:
                        ok, msg = app.add_edge_by_id(u_idx, v_idx, weight)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

            st.subheader("Xóa Cạnh")
            col_del_1, col_del_2 = st.columns(2)
            with col_del_1:
                u_del_label = st.selectbox("Điểm A", list(node_opts.keys()), key="u_del")
            with col_del_2:
                v_del_label = st.selectbox("Điểm B", list(node_opts.keys()), key="v_del")
            
            col_b_del, col_b_clear_edges = st.columns(2)
            with col_b_del:
                if st.button("Xóa cạnh này"):
                    u_idx = node_opts[u_del_label]
                    v_idx = node_opts[v_del_label]
                    ok, msg = app.remove_edge_by_id(u_idx, v_idx)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            with col_b_clear_edges:
                if st.button("Xóa TẤT CẢ các cạnh"):
                    app.clear_all_edges()
                    st.toast("Đã xóa tất cả các cạnh!")
                    st.rerun()

    # --- TAB 3: HIỂN THỊ MA TRẬN ---
    with tab_matrices:
        if not app.nodes:
            st.info("Chưa có dữ liệu đồ thị.")
        else:
            labels = [f"[{i}] {name}" for i, name in enumerate(app.nodes)]
            
            st.subheader("Ma trận kề (Không trọng số)")
            df_adj = pd.DataFrame(app.adj_matrix, index=labels, columns=labels)
            st.dataframe(df_adj, use_container_width=True)

            st.subheader("Ma trận kề có trọng số (Khoảng cách)")
            df_weight = pd.DataFrame(app.weight_matrix, index=labels, columns=labels)
            st.dataframe(df_weight, use_container_width=True)

    # --- TAB 4: THUẬT TOÁN EULER & HAMILTON ---
    with tab_algorithms:
        if not app.nodes:
            st.info("Chưa có dữ liệu đồ thị để chạy thuật toán.")
        else:
            st.subheader("1. Kiểm tra & Tìm Đường đi Euler")
            if st.button("Chạy giải thuật Euler"):
                res = app.solve_euler()
                st.markdown(res)

            st.markdown("---")
            st.subheader("2. Tìm Đường đi Hamilton")
            node_opts = {f"[{i}] {name}": i for i, name in enumerate(app.nodes)}
            
            start_label = st.selectbox("Điểm bắt đầu", list(node_opts.keys()), key="h_start")
            end_label = st.selectbox("Điểm kết thúc", list(node_opts.keys()), key="h_end")
            
            if st.button("Chạy giải thuật Hamilton"):
                s_idx = node_opts[start_label]
                e_idx = node_opts[end_label]
                res = app.solve_hamilton(s_idx, e_idx)
                st.markdown(res)


# ==================== CỘT HIỂN THỊ TRỰC QUAN (VISUALIZER) ====================
with col_visual:
    st.header("📊 Màn hình Kéo thả (Visualizer)")
    
    if not app.nodes:
        st.info("Chưa có điểm nào trong đồ thị. Hãy dùng các công cụ ở cột bên trái để tạo điểm!")
    else:
        # Chuyển đổi dữ liệu sang Node & Edge của streamlit-agraph
        nodes_graph = [
            Node(
                id=str(i), 
                label=f"{name} [{i}]", 
                size=25, 
                color="#4F46E5",
                font={'color': 'white'}
            ) for i, name in enumerate(app.nodes)
        ]
        
        edges_graph = []
        for u in range(len(app.nodes)):
            for v in range(u + 1, len(app.nodes)):
                if app.adj_matrix[u][v] == 1:
                    w = app.weight_matrix[u][v]
                    edges_graph.append(
                        Edge(
                            source=str(u),
                            target=str(v),
                            label=str(w),
                            fontSize=14,
                            color="#64748B"
                        )
                    )

        config = Config(
            width=850,
            height=650,
            directed=False,
            physics=True,  # Bật tính năng vật lý cho phép kéo thả đàn hồi
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#EF4444"
        )

        agraph(nodes=nodes_graph, edges=edges_graph, config=config)
