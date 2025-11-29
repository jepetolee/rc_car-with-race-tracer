#!/usr/bin/env python3
"""
서버 API: 라즈베리 파이에서 수집한 데이터를 받아 학습 수행
Flask 기반 REST API 서버
"""

import os
import pickle
import argparse
import uuid
import numpy as np
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch

# 학습 관련 임포트
from train_with_teacher_forcing import TeacherForcingTrainer
from train_ppo import train as train_dqn_env
from train_imitation_rl import ImitationRLTrainer
from ppo_agent import DQNAgent

app = Flask(__name__)
CORS(app)  # CORS 허용 (라즈베리 파이에서 접근 가능하도록)

# 전역 변수
UPLOAD_FOLDER = 'uploaded_data'
MODEL_FOLDER = 'trained_models'
TEMP_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# 스트리밍 업로드 세션 관리
upload_sessions = {}  # {session_id: {'filename': str, 'file_size': int, 'chunks': {}, 'total_chunks': int}}


@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload_data/init', methods=['POST'])
def upload_data_init():
    """
    스트리밍 업로드 초기화
    
    요청:
    - filename: 파일명
    - file_size: 파일 크기
    - chunk_size: 청크 크기
    - total_chunks: 총 청크 수
    
    응답:
    - session_id: 세션 ID
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        filename = data.get('filename')
        file_size = data.get('file_size')
        chunk_size = data.get('chunk_size')
        total_chunks = data.get('total_chunks')
        
        if not all([filename, file_size is not None, chunk_size is not None, total_chunks is not None]):
            return jsonify({
                'error': 'Missing required fields',
                'received': {
                    'filename': filename,
                    'file_size': file_size,
                    'chunk_size': chunk_size,
                    'total_chunks': total_chunks
                }
            }), 400
        
        # 세션 ID 생성
        session_id = str(uuid.uuid4())
        
        # 세션 정보 저장
        upload_sessions[session_id] = {
            'filename': filename,
            'file_size': file_size,
            'chunk_size': chunk_size,
            'total_chunks': total_chunks,
            'chunks': {},  # {chunk_index: chunk_path}
            'received_chunks': set()
        }
        
        print(f"📥 업로드 세션 시작: {session_id} ({filename}, {file_size / (1024*1024):.2f} MB)")
        
        return jsonify({
            'status': 'success',
            'session_id': session_id
        })
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': f'Server error: {error_msg}'}), 500


@app.route('/api/upload_data/chunk', methods=['POST'])
def upload_data_chunk():
    """
    청크 업로드
    
    요청:
    - session_id: 세션 ID
    - chunk_index: 청크 인덱스
    - chunk: 청크 데이터
    
    응답:
    - status: success
    - received_chunks: 받은 청크 수
    """
    try:
        if 'chunk' not in request.files:
            return jsonify({'error': 'No chunk provided'}), 400
        
        session_id = request.form.get('session_id')
        chunk_index = int(request.form.get('chunk_index'))
        
        if session_id not in upload_sessions:
            return jsonify({'error': 'Invalid session_id'}), 400
        
        session = upload_sessions[session_id]
        chunk = request.files['chunk']
        
        # 청크 저장
        chunk_path = os.path.join(TEMP_FOLDER, f"{session_id}_chunk_{chunk_index}")
        chunk.save(chunk_path)
        
        session['chunks'][chunk_index] = chunk_path
        session['received_chunks'].add(chunk_index)
        
        received = len(session['received_chunks'])
        total = session['total_chunks']
        
        return jsonify({
            'status': 'success',
            'received_chunks': received,
            'total_chunks': total
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_data/finish', methods=['POST'])
def upload_data_finish():
    """
    업로드 완료 및 파일 조립
    
    요청:
    - session_id: 세션 ID
    
    응답:
    - status: success
    - file_path: 저장된 파일 경로
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in upload_sessions:
            return jsonify({'error': 'Invalid session_id'}), 400
        
        session = upload_sessions[session_id]
        received = len(session['received_chunks'])
        total = session['total_chunks']
        
        if received != total:
            return jsonify({'error': f'Missing chunks: {received}/{total}'}), 400
        
        # 파일 조립
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"demos_{timestamp}.pkl"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        print(f"🔨 파일 조립 중: {session['filename']} → {filename}")
        
        with open(filepath, 'wb') as f:
            for i in range(total):
                chunk_path = session['chunks'][i]
                with open(chunk_path, 'rb') as chunk_file:
                    f.write(chunk_file.read())
                # 임시 청크 파일 삭제
                os.remove(chunk_path)
        
        # 세션 삭제
        del upload_sessions[session_id]
        
        # 데이터 검증
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            num_episodes = len(data.get('demonstrations', []))
            total_steps = sum(len(ep.get('states', [])) for ep in data.get('demonstrations', []))
        except Exception as e:
            return jsonify({'error': f'Invalid pickle file: {str(e)}'}), 400
        
        print(f"✅ 파일 조립 완료: {filename} ({num_episodes} 에피소드, {total_steps} 스텝)")
        
        return jsonify({
            'status': 'success',
            'file_path': filepath,
            'filename': filename,
            'num_episodes': num_episodes,
            'total_steps': total_steps
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """
    라즈베리 파이에서 수집한 데이터 업로드
    
    요청:
    - Content-Type: multipart/form-data
    - file: pickle 파일 (human_demos.pkl)
    
    응답:
    - status: success/error
    - file_path: 저장된 파일 경로
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # 파일 크기 확인
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file_size_mb = file_size / (1024 * 1024)
        file.seek(0)  # 다시 처음으로
        
        print(f"📤 파일 업로드 시작: {file.filename} ({file_size_mb:.2f} MB)")
        
        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"demos_{timestamp}.pkl"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 청크 단위로 저장 (대용량 파일 지원)
        chunk_size = 1024 * 1024  # 1MB 청크
        with open(filepath, 'wb') as f:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        
        # 데이터 검증
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            num_episodes = len(data.get('demonstrations', []))
            total_steps = sum(len(ep.get('states', [])) for ep in data.get('demonstrations', []))
        except Exception as e:
            return jsonify({'error': f'Invalid pickle file: {str(e)}'}), 400
        
        return jsonify({
            'status': 'success',
            'file_path': filepath,
            'filename': filename,
            'num_episodes': num_episodes,
            'total_steps': total_steps
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 패치 업로드를 위한 임시 저장소
PATCH_STORAGE = {}  # {session_id: {'patches': [...], 'metadata': {...}}}


@app.route('/api/upload_patch', methods=['POST'])
def upload_patch():
    """
    패치 단위로 데이터 업로드 (16x16 이미지 패치)
    
    요청:
    - session_id: 세션 ID (같은 업로드 세션)
    - patch_index: 패치 인덱스
    - total_patches: 총 패치 수
    - states: 이미지 패치 배열 (16x16 이미지들의 리스트)
    - actions: 액션 배열
    - metadata: 메타데이터 (첫 패치에만)
    
    응답:
    - status: success
    - patch_index: 받은 패치 인덱스
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        patch_index = data.get('patch_index')
        total_patches = data.get('total_patches')
        states = data.get('states')
        actions = data.get('actions')
        metadata = data.get('metadata')
        
        if not session_id:
            return jsonify({'error': 'session_id required'}), 400
        
        # 세션 초기화
        if session_id not in PATCH_STORAGE:
            PATCH_STORAGE[session_id] = {
                'patches': [],
                'metadata': None,
                'total_patches': total_patches
            }
        
        # 패치 저장
        PATCH_STORAGE[session_id]['patches'].append({
            'index': patch_index,
            'states': states,
            'actions': actions
        })
        
        # 메타데이터 저장 (첫 패치)
        if metadata and PATCH_STORAGE[session_id]['metadata'] is None:
            PATCH_STORAGE[session_id]['metadata'] = metadata
        
        return jsonify({
            'status': 'success',
            'patch_index': patch_index,
            'received_patches': len(PATCH_STORAGE[session_id]['patches']),
            'total_patches': total_patches
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/merge_patches', methods=['POST'])
def merge_patches():
    """
    업로드된 패치들을 하나의 파일로 병합
    
    요청:
    - session_id: 세션 ID
    
    응답:
    - status: success
    - file_path: 저장된 파일 경로
    - total_samples: 총 샘플 수
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id or session_id not in PATCH_STORAGE:
            return jsonify({'error': 'Invalid session_id'}), 400
        
        session_data = PATCH_STORAGE[session_id]
        patches = session_data['patches']
        metadata = session_data['metadata']
        
        if len(patches) == 0:
            return jsonify({'error': 'No patches found'}), 400
        
        # 패치들을 인덱스 순으로 정렬
        patches.sort(key=lambda x: x['index'])
        
        # 모든 패치 병합
        all_states = []
        all_actions = []
        
        for patch in patches:
            all_states.extend(patch['states'])
            all_actions.extend(patch['actions'])
        
        # 에피소드 형태로 변환 (단일 에피소드로)
        demonstrations = [{
            'states': all_states,
            'actions': all_actions,
            'rewards': [0.0] * len(all_states),  # 리워드는 0으로 설정
            'dones': [False] * (len(all_states) - 1) + [True],
            'timestamps': []
        }]
        
        # 메타데이터 업데이트
        if metadata:
            metadata['num_episodes'] = 1
            metadata['total_steps'] = len(all_states)
        
        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"demos_patched_{timestamp}.pkl"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'metadata': metadata or {},
                'demonstrations': demonstrations
            }, f)
        
        # 세션 데이터 삭제
        del PATCH_STORAGE[session_id]
        
        return jsonify({
            'status': 'success',
            'file_path': filepath,
            'filename': filename,
            'total_samples': len(all_states),
            'num_patches': len(patches)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train/supervised', methods=['POST'])
def train_supervised():
    """
    Supervised Learning (Teacher Forcing) 학습 시작
    
    요청:
    - file_path: 업로드된 데이터 파일 경로
    - model_path: 사전 학습된 모델 경로 (선택, 없으면 랜덤 초기화)
    - epochs: 학습 에폭 수 (기본: 100)
    - batch_size: 배치 크기 (기본: 64)
    - learning_rate: 학습률 (기본: 3e-4)
    
    응답:
    - status: success
    - model_path: 학습된 모델 경로
    """
    try:
        data = request.json
        file_path = data.get('file_path')
        model_path = data.get('model_path')
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 64)
        learning_rate = data.get('learning_rate', 3e-4)
        
        print(f"📚 Teacher Forcing 학습 요청:")
        print(f"   받은 데이터: {data}")
        print(f"   파일: {file_path}")
        print(f"   에폭: {epochs}")
        print(f"   배치 크기: {batch_size}")
        print(f"   학습률: {learning_rate}")
        
        if not file_path:
            return jsonify({'error': 'file_path is required'}), 400
        
        # 파일 경로 확인 (절대 경로 또는 상대 경로)
        if not os.path.isabs(file_path):
            # 상대 경로인 경우 UPLOAD_FOLDER 기준으로 변환
            file_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file_path))
        
        print(f"   실제 파일 경로: {file_path}")
        print(f"   파일 존재 여부: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            available_files = []
            if os.path.exists(UPLOAD_FOLDER):
                available_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pkl')]
            return jsonify({
                'error': f'File not found: {file_path}',
                'upload_folder': UPLOAD_FOLDER,
                'available_files': available_files[:10]
            }), 400
        
        # 데이터 로드
        with open(file_path, 'rb') as f:
            demo_data = pickle.load(f)
        
        demonstrations = demo_data.get('demonstrations', [])
        if len(demonstrations) == 0:
            return jsonify({'error': 'No demonstrations found'}), 400
        
        # 상태 차원 자동 감지
        state_dim = None
        if len(demonstrations) > 0:
            first_episode = demonstrations[0]
            states = first_episode.get('states', [])
            if len(states) > 0:
                first_state = np.array(states[0])
                if len(first_state.shape) == 1:
                    state_dim = first_state.shape[0]
                else:
                    state_dim = first_state.size
                print(f"📐 상태 차원 자동 감지: {state_dim}")
        
        if state_dim is None:
            return jsonify({'error': 'Could not determine state_dim from demonstrations'}), 400
        
        first_episode = demonstrations[0]
        actions = first_episode.get('actions', [])
        if len(actions) > 0:
            action_dim = int(np.max(actions)) + 1
        else:
            return jsonify({'error': 'Could not determine action_dim from demonstrations'}), 400
        print(f"📐 액션 차원: {action_dim}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            hidden_dim=256,
            latent_dim=256,
        )
        loaded_model_path = None
        if model_path and os.path.exists(model_path):
            try:
                agent.load(model_path, strict=False)
                loaded_model_path = model_path
                print(f"✅ 기존 모델 로드 완료: {model_path}")
            except Exception as e:
                print(f"⚠️  모델 로드 실패: {e}. 랜덤 초기화로 진행합니다.")
                model_path = None
        else:
            if model_path:
                print(f"⚠️  모델 파일을 찾을 수 없습니다: {model_path}")
            model_path = None
        
        trainer = TeacherForcingTrainer(agent, demonstrations, device=device, lr=learning_rate)
        model_path = os.path.join(MODEL_FOLDER, f"pretrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        
        trainer.pretrain(
            epochs=epochs,
            batch_size=batch_size,
            save_path=model_path,
            verbose=True
        )
        
        return jsonify({
            'status': 'success',
            'model_path': model_path,
            'epochs': epochs,
            'state_dim': state_dim,
            'loaded_model': loaded_model_path
        })
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ Teacher Forcing 학습 실패:")
        print(error_trace)
        return jsonify({
            'error': error_msg,
            'traceback': error_trace
        }), 500


@app.route('/api/train/imitation_rl', methods=['POST'])
def train_imitation_rl_api():
    """
    Imitation Learning via Reinforcement Learning 학습 시작
    
    요청:
    - file_path: 업로드된 데모 데이터 파일 경로
    - model_path: 사전 학습된 모델 경로 (선택)
    - epochs: 학습 에폭 수 (기본: 100)
    - batch_size: 배치 크기 (기본: 64)
    - learning_rate: 학습률 (기본: 3e-4)
    
    응답:
    - status: success
    - model_path: 학습된 모델 경로
    - final_match_rate: 최종 일치율
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        file_path = data.get('file_path')
        model_path = data.get('model_path')
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 64)
        learning_rate = data.get('learning_rate', 3e-4)
        
        print(f"📚 Imitation RL 학습 요청:")
        print(f"   받은 데이터: {data}")
        print(f"   파일: {file_path}")
        print(f"   에폭: {epochs}")
        print(f"   배치 크기: {batch_size}")
        print(f"   학습률: {learning_rate}")
        
        if not file_path:
            return jsonify({
                'error': 'file_path is required',
                'received_data': data
            }), 400
        
        # 파일 경로 확인 (절대 경로 또는 상대 경로)
        if not os.path.isabs(file_path):
            # 상대 경로인 경우 UPLOAD_FOLDER 기준으로 변환
            file_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file_path))
        
        print(f"   실제 파일 경로: {file_path}")
        print(f"   파일 존재 여부: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            # 파일을 찾을 수 없을 때 가능한 파일 목록 표시
            available_files = []
            if os.path.exists(UPLOAD_FOLDER):
                available_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pkl')]
            return jsonify({
                'error': f'File not found: {file_path}',
                'upload_folder': UPLOAD_FOLDER,
                'available_files': available_files[:10]  # 최대 10개만 표시
            }), 400
        
        # model_path 처리 (상대 경로인 경우 여러 위치 확인)
        if model_path:
            # 상대 경로인 경우 프로젝트 루트, MODEL_FOLDER, 현재 디렉토리 확인
            if not os.path.isabs(model_path):
                # 여러 가능한 경로 확인
                possible_paths = [
                    model_path,  # 현재 디렉토리 기준
                    os.path.join(MODEL_FOLDER, model_path),  # MODEL_FOLDER 기준
                    os.path.join(MODEL_FOLDER, os.path.basename(model_path)),  # 파일명만 사용
                    os.path.join('.', model_path),  # 현재 디렉토리 명시
                ]
                
                found = False
                for candidate in possible_paths:
                    if os.path.exists(candidate):
                        model_path = candidate
                        found = True
                        break
                
                if not found:
                    print(f"⚠️  지정된 모델 파일을 찾을 수 없습니다: {model_path}")
                    print(f"   시도한 경로들: {possible_paths}")
                    model_path = None
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   디바이스: {device}")
        trainer = ImitationRLTrainer(
            demos_path=file_path,
            model_path=model_path if model_path and os.path.exists(model_path) else None,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        updates_per_epoch = data.get('updates_per_epoch', 1000)
        
        try:
            print("🚀 학습 시작...")
            model_path = trainer.train(
                epochs=epochs,
                updates_per_epoch=updates_per_epoch
            )
            print(f"✅ 학습 완료: {model_path}")
        except Exception as e:
            import traceback
            error_msg = f"학습 실행 실패: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return jsonify({'error': error_msg}), 500
        
        # 최종 평가
        try:
            final_match_rate = trainer.evaluate()
            print(f"📊 최종 일치율: {final_match_rate:.2%}")
        except Exception as e:
            print(f"⚠️  평가 실패: {e}")
            final_match_rate = 0.0
        
        return jsonify({
            'status': 'success',
            'model_path': model_path,
            'final_match_rate': float(final_match_rate),
            'epochs': epochs
        })
    
    except Exception as e:
        import traceback
        error_msg = f"서버 오류: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@app.route('/api/train/ppo', methods=['POST'])
def train_ppo_api():
    """
    TRM-DQN 강화학습 시작
    
    요청:
    - env_type: 환경 타입 (carracing/sim/real)
    - epsilon/학습률 등 DQN 하이퍼파라미터
    
    응답:
    - status: success
    - model_path: 학습된 모델 경로
    """
    try:
        data = request.json or {}
        config = argparse.Namespace(
            env_type=data.get('env_type', 'carracing'),
            state_dim=data.get('state_dim', 784),
            action_dim=data.get('action_dim', 5),
            hidden_dim=data.get('hidden_dim', 256),
            latent_dim=data.get('latent_dim', 256),
            n_deep_loops=data.get('n_deep_loops', 2),
            n_latent_loops=data.get('n_latent_loops', 2),
            gamma=data.get('gamma', 0.99),
            learning_rate=data.get('learning_rate', 3e-4),
            batch_size=data.get('batch_size', 128),
            replay_buffer=data.get('replay_buffer', 200_000),
            target_update_interval=data.get('target_update_interval', 2000),
            max_grad_norm=data.get('max_grad_norm', 1.0),
            max_episodes=data.get('max_episodes', 1000),
            max_episode_steps=data.get('max_episode_steps', 1000),
            eps_start=data.get('eps_start', 1.0),
            eps_end=data.get('eps_end', 0.05),
            eps_decay=data.get('eps_decay', 300_000),
            save_dir=MODEL_FOLDER,
            save_interval=data.get('save_interval', 50),
            use_tensorboard=False,
        )
        train_dqn_env(config)
        latest_models = sorted(
            [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
        )
        model_path = os.path.join(MODEL_FOLDER, latest_models[-1]) if latest_models else None
        return jsonify({
            'status': 'success',
            'model_path': model_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/latest', methods=['GET'])
def get_latest_model():
    """
    최신 모델 다운로드
    
    응답:
    - 모델 파일 (.pth)
    """
    try:
        # MODEL_FOLDER에서 최신 모델 찾기
        model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
        if not model_files:
            return jsonify({'error': 'No model found'}), 404
        
        # 최신 파일 선택 (이름 기준)
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(MODEL_FOLDER, latest_model)
        
        return send_file(
            model_path,
            as_attachment=True,
            download_name=latest_model
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/list', methods=['GET'])
def list_models():
    """
    사용 가능한 모델 목록 조회
    
    응답:
    - models: 모델 파일 목록
    """
    try:
        model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
        model_info = []
        
        for model_file in sorted(model_files, reverse=True):
            model_path = os.path.join(MODEL_FOLDER, model_file)
            stat = os.stat(model_path)
            model_info.append({
                'filename': model_file,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return jsonify({
            'models': model_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/download/<filename>', methods=['GET'])
def download_model(filename):
    """
    특정 모델 다운로드
    
    Args:
        filename: 다운로드할 모델 파일명 (예: pretrained_20251129_190816.pth)
    
    응답:
        - 모델 파일 (.pth)
    """
    try:
        # 보안: 경로 탐색 방지
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        model_path = os.path.join(MODEL_FOLDER, filename)
        
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model not found: {filename}'}), 404
        
        if not filename.endswith('.pth'):
            return jsonify({'error': 'Invalid file type. Only .pth files are allowed'}), 400
        
        return send_file(
            model_path,
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/inference', methods=['POST'])
def inference():
    """
    실시간 추론 (선택 사항)
    
    요청:
    - state: 256차원 상태 벡터 (16x16 이미지)
    - model_path: 사용할 모델 경로 (선택, 기본: 최신 모델)
    
    응답:
    - action: 추론된 액션 (0-4)
    - q_values: 각 액션의 Q값
    """
    try:
        data = request.json
        state = data.get('state')
        model_path = data.get('model_path')
        
        if state is None:
            return jsonify({'error': 'No state provided'}), 400
        
        # 모델 로드
        if model_path is None:
            # 최신 모델 사용
            model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
            if not model_files:
                return jsonify({'error': 'No model found'}), 404
            model_path = os.path.join(MODEL_FOLDER, sorted(model_files)[-1])
        
        # 에이전트 로드 및 추론
        state_dim = len(state)
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=5,
            device='cpu',
        )
        agent.load(model_path, strict=False)
        
        state_vec = np.array(state, dtype=np.float32)
        action = agent.act_greedy(state_vec)
        q_values = agent.predict(np.expand_dims(state_vec, axis=0))[0]

        return jsonify({
            'action': int(action),
            'q_values': [float(q) for q in q_values]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# QR 데이터 저장 디렉토리
QR_DATA_FOLDER = 'qr_dataset'
os.makedirs(QR_DATA_FOLDER, exist_ok=True)
os.makedirs(os.path.join(QR_DATA_FOLDER, 'qr_present'), exist_ok=True)
os.makedirs(os.path.join(QR_DATA_FOLDER, 'qr_absent'), exist_ok=True)


@app.route('/api/upload_qr_data', methods=['POST'])
def upload_qr_data():
    """
    QR 데이터 업로드 (이미지 배치)
    
    요청:
    - images: base64 인코딩된 이미지 리스트
    - labels: 라벨 리스트 (0: QR 없음, 1: QR 있음)
    - metadata: 메타데이터 (선택)
    
    응답:
    - status: success
    - saved_count: 저장된 이미지 수
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.json
        images_base64 = data.get('images', [])
        labels = data.get('labels', [])
        metadata = data.get('metadata', {})
        
        if len(images_base64) != len(labels):
            return jsonify({'error': 'Images and labels count mismatch'}), 400
        
        if len(images_base64) == 0:
            return jsonify({'error': 'No images provided'}), 400
        
        import base64
        import cv2
        
        saved_count = 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i, (img_base64, label) in enumerate(zip(images_base64, labels)):
            try:
                # base64 디코딩
                img_bytes = base64.b64decode(img_base64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # 파일명 생성
                label_str = 'qr_present' if label == 1 else 'qr_absent'
                filename = f"{label_str}_{timestamp}_{i:04d}.png"
                filepath = os.path.join(QR_DATA_FOLDER, label_str, filename)
                
                # 이미지 저장
                cv2.imwrite(filepath, img)
                saved_count += 1
                
            except Exception as e:
                print(f"⚠️  이미지 {i} 저장 실패: {e}")
                continue
        
        print(f"📥 QR 데이터 업로드: {saved_count}/{len(images_base64)}장 저장 완료")
        
        return jsonify({
            'status': 'success',
            'saved_count': saved_count,
            'total_count': len(images_base64),
            'metadata': metadata
        })
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': f'Server error: {error_msg}'}), 500


def main():
    parser = argparse.ArgumentParser(description='RC Car 학습 서버 API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='서버 호스트 (기본: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='서버 포트 (기본: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드')
    
    args = parser.parse_args()
    
    print(f"🚀 서버 시작: http://{args.host}:{args.port}")
    print(f"📁 업로드 폴더: {UPLOAD_FOLDER}")
    print(f"📁 모델 폴더: {MODEL_FOLDER}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

