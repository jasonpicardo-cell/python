/* Shared engine for the Nifty guide + cheat sheet: Black-Scholes, payoff and Greek charts drawn as inline SVG. */
const NF={S:24500,IV:0.13,R:0.065,LOT:65};
const Nc=x=>{const t=1/(1+0.2316419*Math.abs(x)),d=0.3989423*Math.exp(-x*x/2),p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));return x>0?1-p:p};
const np=x=>Math.exp(-x*x/2)/2.5066283;
const ATM=()=>Math.round(NF.S/100)*100;
const ivK=K=>NF.IV*Math.min(2,Math.max(.6,1+2.2*(NF.S-K)/NF.S));          // simple put skew
function d12(S,K,T,v){const s=v*Math.sqrt(T),a=(Math.log(S/K)+(NF.R+v*v/2)*T)/s;return[a,a-s]}
function bs(cp,S,K,T,v){if(T<=1e-7)return cp=='c'?Math.max(S-K,0):Math.max(K-S,0);const[a,b]=d12(S,K,T,v),D=K*Math.exp(-NF.R*T);return cp=='c'?S*Nc(a)-D*Nc(b):D*Nc(-b)-S*Nc(-a)}
function gk(cp,S,K,T,v){if(cp=='f')return{d:1,g:0,t:0,v:0};T=Math.max(T,1e-5);const[a,b]=d12(S,K,T,v),q=Math.sqrt(T),D=K*Math.exp(-NF.R*T),c=cp=='c';
 return{d:c?Nc(a):Nc(a)-1,g:np(a)/(S*v*q),v:S*np(a)*q/100,t:(-S*np(a)*v/(2*q)-(c?NF.R*D*Nc(b):-NF.R*D*Nc(-b)))/365}}
let UID=0;
function ticks(a,b,n){if(a>b)[a,b]=[b,a];const r=(b-a)/n,p=10**Math.floor(Math.log10(r)),e=r/p,s=p*(e>=7.5?10:e>=3.5?5:e>=1.5?2:1),o=[];for(let v=Math.ceil(a/s)*s;v<=b+s*1e-6;v+=s)o.push(+v.toFixed(6));return o}
const f0=v=>Math.abs(v)>=1e4?(v/1e3).toFixed(1).replace('.0','')+'k':(+v.toFixed(4)).toString();
function plot(S,o={}){
 const W=600,H=o.h||230,L=56,R=12,T=10,B=30,X=S.flatMap(s=>s.x).concat((o.bars||[]).map(b=>b.x)),Y=S.flatMap(s=>s.y).concat((o.bars||[]).map(b=>b.y));
 let x0=o.x0??Math.min(...X),x1=o.x1??Math.max(...X),y0=o.y0??Math.min(...Y),y1=o.y1??Math.max(...Y);
 if(o.zero){y0=Math.min(y0,0);y1=Math.max(y1,0)}const pd=(y1-y0||1)*.08;y0-=pd;y1+=pd;if(o.bars){const bw=(x1-x0)/o.bars.length;x0-=bw/2;x1+=bw/2}
 const sx=x=>L+(x-x0)/(x1-x0)*(W-L-R),sy=y=>T+(y1-y)/(y1-y0)*(H-T-B);let g='';
 for(const t of ticks(x0,x1,6))g+=`<line class=gr x1=${sx(t)} x2=${sx(t)} y1=${T} y2=${H-B} /><text class=tx x=${sx(t)} y=${H-B+14} text-anchor=middle>${(o.fx||f0)(t)}</text>`;
 for(const t of ticks(y0,y1,5))g+=`<line class=gr x1=${L} x2=${W-R} y1=${sy(t)} y2=${sy(t)} /><text class=tx x=${L-6} y=${sy(t)+4} text-anchor=end>${(o.fy||f0)(t)}</text>`;
 if(y0<0&&y1>0)g+=`<line class=zl x1=${L} x2=${W-R} y1=${sy(0)} y2=${sy(0)} />`;
 (o.vx||[]).forEach(v=>g+=`<line class=vx x1=${sx(v)} x2=${sx(v)} y1=${T} y2=${H-B} />`);
 if(o.bars){const bw=(sx(o.bars[1].x)-sx(o.bars[0].x))*.7;o.bars.forEach(b=>{const z=sy(0),y=sy(b.y);g+=`<rect class=${b.y>=0?'bp':'bn'} x=${sx(b.x)-bw/2} y=${Math.min(z,y)} width=${bw} height=${Math.abs(z-y)} />`})}
 for(const s of S){const pts=s.x.map((x,i)=>`${sx(x).toFixed(1)},${sy(s.y[i]).toFixed(1)}`).join(' ');
  if(s.fill){const id='q'+(++UID),z=sy(Math.max(y0,Math.min(0,y1))),poly=`<polygon points="${sx(s.x[0])},${z} ${pts} ${sx(s.x.at(-1))},${z}"`;
   g+=`<clipPath id=${id}a><rect width=${W} height=${z} /></clipPath><clipPath id=${id}b><rect y=${z} width=${W} height=${H} /></clipPath>${poly} class=fp clip-path="url(#${id}a)"/>${poly} class=fn clip-path="url(#${id}b)"/>`}
  g+=`<polyline class="ln k${s.k||1}${s.dash?' ds':''}" points="${pts}"/>`}
 (o.notes||[]).forEach(n=>g+=`<text class=nt x=${sx(n[0])} y=${sy(n[1])} text-anchor=middle>${n[2]}</text>`);
 const lg=S.filter(s=>s.lab).map(s=>`<span class=k${s.k||1}>${s.dash?'┅':'━'} ${s.lab}</span>`).join('')+(o.lg||'');
 return `<svg viewBox="0 0 ${W} ${H}" role=img aria-label="${o.al||'chart'}">${g}</svg><div class=lgd>${o.yl?`<i>y: ${o.yl}</i>`:''}${o.xl?`<i>x: ${o.xl}</i>`:''}${lg}</div>`}
/* legs: "c0:-1 p-300:1 f:1 c200:1@35" -> type, strike offset from ATM, qty, days to expiry (default 7) */
function legs(s){const A=ATM();return s.trim().split(/\s+/).map(t=>{const m=t.match(/^([cpf])([+-]?\d*):(-?\d+)(?:@(\d+))?$/),l={cp:m[1],K:A+(+m[2]||0),q:+m[3],d:+(m[4]||7)};l.v=ivK(l.K);l.p=l.cp=='f'?NF.S:bs(l.cp,NF.S,l.K,l.d/365,l.v);return l})}
const lv=(l,S,d)=>l.cp=='f'?S:bs(l.cp,S,l.K,Math.max(d,0)/365,l.v);
const RS=v=>'₹'+Math.round(v*NF.LOT).toLocaleString('en-IN');
function pay(el){const L=legs(el.dataset.l),r=+(el.dataset.r||.06),tE=Math.min(...L.map(l=>l.d)),n=161,xs=[],ex=[],nw=[];
 for(let i=0;i<n;i++){const S=NF.S*(1-r+2*r*i/(n-1));xs.push(S);ex.push(L.reduce((a,l)=>a+l.q*(lv(l,S,l.d-tE)-l.p),0));nw.push(L.reduce((a,l)=>a+l.q*(lv(l,S,l.d)-l.p),0))}
 const fa=k=>L.reduce((a,l)=>a+l.q*(lv(l,NF.S*k,l.d-tE)-l.p),0),fL=fa(.6),fH=fa(1.4),e0=Math.max(...ex),e1=Math.min(...ex),tol=Math.max(5,.15*(e0-e1)),oH=fH>e0+tol||fL>e0+tol,oL=fH<e1-tol||fL<e1-tol;
 const mx=oH?1/0:Math.max(e0,fL,fH),mn=oL?-1/0:Math.min(e1,fL,fH),U=v=>!isFinite(v),be=[];
 for(let i=1;i<n;i++)if(ex[i-1]*ex[i]<0)be.push(xs[i-1]+(xs[i]-xs[i-1])*ex[i-1]/(ex[i-1]-ex[i]));
 const net=-L.filter(l=>l.cp!='f').reduce((a,l)=>a+l.q*l.p,0),G=L.reduce((a,l)=>{const g=gk(l.cp,NF.S,l.K,l.d/365,l.v);for(const k in g)a[k]+=l.q*g[k];return a},{d:0,g:0,t:0,v:0}),P=v=>v.toFixed(1);
 const leg=L.map(l=>`${l.q>0?'+':''}${l.q} ${l.cp=='f'?'FUT':l.K+(l.cp=='c'?'CE':'PE')+(l.cp!='f'?` @${l.p.toFixed(1)}`:'')}${l.d!=7&&l.cp!='f'?` (${l.d}d)`:''}`).join(' · ');
 el.innerHTML=plot([{x:xs,y:ex,k:1,fill:1,lab:`at ${tE}-day expiry`},{x:xs,y:nw,k:2,dash:1,lab:'today (T+0)'}],{zero:1,vx:[NF.S],h:190,fx:v=>v.toFixed(0),al:'payoff'})+
 `<div class=ps><div class=lg2>${leg}</div><b>${net>=0?'Credit':'Debit'}</b> ${P(Math.abs(net))} (${RS(Math.abs(net))}) · <b>Max P</b> ${U(mx)?'open':P(mx)+' ('+RS(mx)+')'} · <b>Max L</b> ${U(mn)?'open':P(-mn)+' ('+RS(-mn)+')'} · <b>BE</b> ${be.map(b=>b.toFixed(0)).join(' / ')||'n/a'}<br>Δ ${G.d.toFixed(2)} · Γ ${(G.g*1e4).toFixed(1)}e-4 · Θ ${G.t.toFixed(1)}/day · Vega ${G.v.toFixed(1)}/vol pt <span class=mu>(points; ₹ per ${NF.LOT}-qty lot)</span></div>`}
const rng=(a,b,n)=>Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
const CH={
 delta:()=>{const K=ATM(),xs=rng(NF.S*.95,NF.S*1.05,121);return plot([30,7,1].map((d,i)=>({x:xs,y:xs.map(S=>gk('c',S,K,d/365,NF.IV).d),k:i+1,lab:`${d} DTE`})),{vx:[K],fx:v=>v.toFixed(0),yl:'call delta',xl:'spot',al:'delta'})},
 gamma:()=>{const K=ATM(),xs=rng(NF.S*.95,NF.S*1.05,121);return plot([30,7,1].map((d,i)=>({x:xs,y:xs.map(S=>gk('c',S,K,d/365,NF.IV).g*1e4),k:i+1,lab:`${d} DTE`})),{vx:[K],fx:v=>v.toFixed(0),yl:'gamma ×10⁻⁴ (delta change per point)',xl:'spot'})},
 decay:()=>{const K=ATM(),xs=rng(60,0,121);return plot([{x:xs,y:xs.map(d=>bs('c',NF.S,K,d/365,NF.IV)+bs('p',NF.S,K,d/365,NF.IV)),k:1,fill:0,lab:'ATM straddle value'},{x:xs,y:xs.map(d=>.8*NF.S*NF.IV*Math.sqrt(d/365)),k:3,dash:1,lab:'0.8·S·σ·√T shortcut'}],{x0:60,x1:0,yl:'points',xl:'days to expiry (time runs →)'})},
 theta:()=>{const K=ATM(),xs=rng(45,.25,120);return plot([{x:xs,y:xs.map(d=>-(gk('c',NF.S,K,d/365,NF.IV).t+gk('p',NF.S,K,d/365,NF.IV).t)),k:2,lab:'ATM straddle theta (pts/day)'},{x:xs,y:xs.map(d=>-(gk('c',NF.S,K+400,d/365,ivK(K+400)).t+gk('p',NF.S,K-400,d/365,ivK(K-400)).t)),k:1,lab:'±400 strangle theta'}],{x0:45,x1:0,yl:'decay per calendar day',xl:'days to expiry'})},
 gv:()=>{const K=ATM(),xs=rng(60,.5,120),g=xs.map(d=>gk('c',NF.S,K,d/365,NF.IV).g),v=xs.map(d=>gk('c',NF.S,K,d/365,NF.IV).v),mg=Math.max(...g),mv=Math.max(...v);
  return plot([{x:xs,y:g.map(a=>a/mg*100),k:2,lab:'gamma (% of its peak)'},{x:xs,y:v.map(a=>a/mv*100),k:1,lab:'vega (% of its peak)'}],{x0:60,x1:0,yl:'relative size',xl:'days to expiry'})},
 smile:()=>{const xs=rng(-.08,.08,81),f=(m,b,s,c)=>(b-s*m+c*m*m)*100;return plot([{x:xs.map(m=>NF.S*Math.exp(m)),y:xs.map(m=>f(m,.13,.3,3)),k:1,lab:'calm: smirk'},{x:xs.map(m=>NF.S*Math.exp(m)),y:xs.map(m=>f(m,.19,.75,5)),k:2,lab:'sell-off: level up, skew steeper'}],{vx:[NF.S],fx:v=>v.toFixed(0),yl:'implied vol %',xl:'strike'})},
 cone:()=>{const xs=rng(0,30,61),s=NF.IV;return plot([1,-1,2,-2].map((k,i)=>({x:xs,y:xs.map(d=>NF.S*Math.exp(k*s*Math.sqrt(d/365))),k:i<2?1:2,dash:i>1,lab:i==0?'±1σ (68%)':i==2?'±2σ (95%)':''})),{yl:`Nifty, IV ${Math.round(s*100)}%`,xl:'calendar days ahead'})},
 cvd:()=>{const x=[...Array(13).keys()],p=[100,103,101,106,104,109,107,111,109,112,108,104,101],c=[0,4,3,8,6,10,9,10,8,9,5,1,-3];
  return plot([{x,y:p.map(v=>(v-100)*20),k:1,lab:'price (futures)'},{x,y:c.map(v=>v*14),k:2,lab:'CVD'}],{yl:'scaled',xl:'bars',notes:[[7,232,'price HH'],[7,155,'CVD LH']],al:'cvd divergence'})},
 gex:()=>{const A=ATM(),v=[-2,-3,-4,-2,1,3,6,8,5,2,-1,-3,-5,-4,-2];return plot([],{bars:v.map((y,i)=>({x:A-700+i*100,y})),fx:t=>t.toFixed(0),yl:'net writer gamma by strike (illustrative)',xl:'strike',vx:[A],h:200})}
};
function vixTable(el){const r=[10,12,14,16,18,20,25,30,40];el.innerHTML='<tr><th>VIX</th><th>1-day 1σ</th><th>1-week 1σ</th><th>1-month 1σ</th><th>ATM weekly straddle* (7d)</th></tr>'+r.map(v=>{const s=v/100,f=k=>`${(s*Math.sqrt(k)*100).toFixed(2)}% ≈ ${Math.round(NF.S*s*Math.sqrt(k))}`;return`<tr><td>${v}</td><td>${f(1/252)}</td><td>${f(1/52)}</td><td>${f(1/12)}</td><td>≈ ${Math.round(.8*NF.S*s*Math.sqrt(7/365))}</td></tr>`}).join('')}
function calc(){const g=id=>+document.getElementById(id).value,S=g('cS'),v=g('cV')/100,d=g('cD'),o=document.getElementById('cO');if(!o)return;
 const st=.8*S*v*Math.sqrt(d/365),dm=S*v/Math.sqrt(252),G=.4/(S*v*Math.sqrt(Math.max(d,.05)/365));
 o.innerHTML=`ATM straddle ≈ <b>${st.toFixed(0)}</b> pts · market-implied ±1σ to expiry ≈ <b>${(S*v*Math.sqrt(d/365)).toFixed(0)}</b> pts · daily 1σ (break-even move for long gamma) ≈ <b>${dm.toFixed(0)}</b> pts · ATM gamma ≈ <b>${(G*1e4).toFixed(1)}e-4</b> (a 100-pt move shifts ATM delta by ${(G*100).toFixed(2)}) · ATM vega ≈ <b>${(.4*S*Math.sqrt(d/365)/100).toFixed(1)}</b> pts per vol pt per option`}
function mini(el){const L=legs(el.dataset.m),r=.05,tE=Math.min(...L.map(l=>l.d)),n=61,y=[];for(let i=0;i<n;i++){const S=NF.S*(1-r+2*r*i/(n-1));y.push(L.reduce((a,l)=>a+l.q*(lv(l,S,l.d-tE)-l.p),0))}
 const lo=Math.min(...y,0),hi=Math.max(...y,0),sy=v=>3+(hi-v)/(hi-lo||1)*30,pts=y.map((v,i)=>`${(i*90/(n-1)).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
 el.innerHTML=`<svg viewBox="0 0 90 36" style="width:90px;height:36px"><line class=zl x1=0 x2=90 y1=${sy(0)} y2=${sy(0)} /><line class=vx x1=45 x2=45 y1=0 y2=36 /><polyline class="ln k1" style="stroke-width:1.8" points="${pts}"/></svg>`}
function renderAll(){document.querySelectorAll('[data-m]').forEach(mini);document.querySelectorAll('[data-l]').forEach(pay);document.querySelectorAll('[data-c]').forEach(e=>e.innerHTML=CH[e.dataset.c]());document.querySelectorAll('[data-vix]').forEach(vixTable);calc()}
function setG(){const g=id=>+document.getElementById(id).value;NF.S=g('gS');NF.IV=g('gV')/100;NF.LOT=g('gL');renderAll()}
document.addEventListener('DOMContentLoaded',()=>{document.querySelectorAll('.thm').forEach(b=>b.onclick=()=>{const r=document.documentElement,d=r.dataset.theme=='dark'||(!r.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);r.dataset.theme=d?'light':'dark'});renderAll()});
