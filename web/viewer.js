import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';

export function createViewer(element,data){
  const scene=new THREE.Scene();
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  const toolbar=document.createElement('div');toolbar.className='viewer-toolbar';
  toolbar.innerHTML=`<span>${new Intl.NumberFormat('zh-CN').format(data.total_points||data.positions.length/3)} points</span><label>显示 <select data-view="color"><option value="rgb">RGB</option><option value="solid">几何灰</option></select></label><label>点大小 <input data-view="size" type="range" min="1" max="7" step=".5" value="2"></label><button class="button small" data-view="grid">网格</button><button class="button small" data-view="reset">聚焦</button><button class="button small" data-view="full">全屏</button>`;
  element.appendChild(toolbar);
  element.appendChild(renderer.domElement);
  const camera=new THREE.PerspectiveCamera(45,1,0.0001,1000);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.Float32BufferAttribute(data.positions,3));
  geo.setAttribute('color',new THREE.Float32BufferAttribute(data.colors,3));
  geo.computeBoundingBox();
  const center=geo.boundingBox.getCenter(new THREE.Vector3());
  const size=Math.max(geo.boundingBox.getSize(new THREE.Vector3()).length(),0.01);
  let object,material;
  if(data.indices){geo.setIndex(data.indices);geo.computeVertexNormals();material=new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:.85});object=new THREE.Mesh(geo,material);scene.add(new THREE.HemisphereLight(0xe3f3ff,0x35434e,2));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(1,2,-1);scene.add(light);}
  else{material=new THREE.PointsMaterial({size:size/450,vertexColors:true,sizeAttenuation:true});object=new THREE.Points(geo,material);}
  // Camera frame has y down. Rotate only visualization; retain downloaded metric geometry.
  const group=new THREE.Group();group.add(object);group.rotation.x=Math.PI;scene.add(group);
  center.applyAxisAngle(new THREE.Vector3(1,0,0),Math.PI);
  const reset=()=>{controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(size*.45,size*.25,size*1.25));camera.up.set(0,1,0);controls.update()};reset();
  camera.near=size/1000;camera.far=size*100;camera.updateProjectionMatrix();
  const grid=new THREE.GridHelper(size*2,20,0x344e60,0x213241);grid.position.copy(center);grid.position.y-=size*.4;scene.add(grid);
  toolbar.querySelector('[data-view="reset"]').onclick=reset;
  toolbar.querySelector('[data-view="grid"]').onclick=()=>grid.visible=!grid.visible;
  toolbar.querySelector('[data-view="full"]').onclick=()=>element.requestFullscreen?.();
  const sizeInput=toolbar.querySelector('[data-view="size"]');sizeInput.oninput=()=>{if(object.isPoints)material.size=size/900*Number(sizeInput.value)};
  const colorInput=toolbar.querySelector('[data-view="color"]');colorInput.onchange=()=>{if(object.isPoints){material.vertexColors=colorInput.value==='rgb';material.color.set(colorInput.value==='rgb'?0xffffff:0xc4ced8);material.needsUpdate=true}};
  const resize=()=>{renderer.setSize(element.clientWidth,element.clientHeight);camera.aspect=element.clientWidth/element.clientHeight;camera.updateProjectionMatrix();};
  const observer=new ResizeObserver(resize);observer.observe(element);resize();
  let raf;const draw=()=>{raf=requestAnimationFrame(draw);controls.update();renderer.render(scene,camera);};draw();
  return ()=>{cancelAnimationFrame(raf);observer.disconnect();controls.dispose();geo.dispose();material.dispose();grid.geometry.dispose();grid.material.dispose();renderer.dispose();renderer.domElement.remove();toolbar.remove();};
}
